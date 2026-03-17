/**
 * Standalone test for the frames.ts prefetch / decode-ahead logic.
 *
 * Tests the exact functions copied from the production code to verify:
 *   1. Parallel network fetches  – 3 chunks prefetched from the server simultaneously
 *   2. Single decode gate        – only 1 chunk decoded at a time (no eviction of current)
 *   3. Correct handoff           – frame at chunk boundary is available via activeChunkRequest
 *   4. No double-fetch           – ensureChunkFetched is idempotent
 *   5. Race safety               – startNextChunkDecode is a no-op when activeChunkRequest is set
 */

let passed = 0;
let failed = 0;

function assert(condition, message) {
    if (condition) {
        console.log(`  ✓ ${message}`);
        passed++;
    } else {
        console.error(`  ✗ FAIL: ${message}`);
        failed++;
    }
}

// ---------------------------------------------------------------------------
// Minimal mock infrastructure
// ---------------------------------------------------------------------------

function makeProvider(chunkCount = 20) {
    const cached = new Set();
    const cleanupCalls = [];
    const decodeCalls = [];

    return {
        isChunkCached: (idx) => cached.has(idx),
        markCached: (idx) => cached.add(idx),
        cleanup: (n) => cleanupCalls.push(n),
        cleanupCalls,
        decodeCalls,
        // Simulates requestDecodeBlock: immediately calls onDecodeAll after a microtask
        requestDecodeBlock: (chunk, chunkIndex, frameNumbers, onDecode, onDecodeAll, onReject) => {
            decodeCalls.push(chunkIndex);
            // simulate async decode completing
            Promise.resolve().then(() => {
                cached.add(chunkIndex);
                onDecodeAll();
            });
        },
    };
}

function makeCache(provider, chunkCount = 20, chunkSize = 36) {
    const fetchCalls = [];
    const cache = {
        activeChunkRequest: null,
        rawChunkFetchCache: new Map(),
        // Simulates getChunk: resolves after a microtask with a fake ArrayBuffer
        getChunk: (chunkIndex) => {
            fetchCalls.push(chunkIndex);
            return Promise.resolve(new ArrayBuffer(8));
        },
        fetchCalls,
    };

    // The production functions, verbatim from frames.ts (adapted to use local vars)
    const jobID = 'job1';
    const frameDataCache = { [jobID]: cache };
    const segmentFrameNumbers = Array.from({ length: chunkCount * chunkSize }, (_, i) => i);
    const meta = { chunkCount };
    const decodeForward = true;
    const decodedBlocksCacheSize = 7;

    const ensureChunkFetched = (targetChunk) => {
        if (!(jobID in frameDataCache)) return;
        if (targetChunk < 0 || targetChunk >= meta.chunkCount) return;
        if (provider.isChunkCached(targetChunk)) return;
        const fetchCache = frameDataCache[jobID].rawChunkFetchCache;
        if (fetchCache.has(targetChunk)) return;
        const fp = frameDataCache[jobID].getChunk(targetChunk);
        fetchCache.set(targetChunk, fp);
        fp.catch(() => { frameDataCache[jobID]?.rawChunkFetchCache.delete(targetChunk); });
    };

    const startNextChunkDecode = (targetChunk) => {
        if (!(jobID in frameDataCache)) return;
        const c = frameDataCache[jobID];
        if (c.activeChunkRequest) return;
        const fetchPromise = c.rawChunkFetchCache.get(targetChunk);
        if (!fetchPromise) return;

        c.activeChunkRequest = new Promise((resolveForward) => {
            const done = () => {
                resolveForward();
                if (jobID in frameDataCache) {
                    frameDataCache[jobID].activeChunkRequest = null;
                    frameDataCache[jobID].rawChunkFetchCache.delete(targetChunk);
                }
            };
            fetchPromise.then((chunk) => {
                if (!(jobID in frameDataCache)) { resolveForward(); return; }
                provider.cleanup(1);
                provider.requestDecodeBlock(
                    chunk, targetChunk,
                    segmentFrameNumbers.slice(targetChunk * chunkSize, (targetChunk + 1) * chunkSize),
                    () => {}, done, done,
                );
            }).catch(() => {
                if (jobID in frameDataCache) frameDataCache[jobID].activeChunkRequest = null;
                resolveForward();
            });
        });
    };

    // Simulates what happens when a cached frame is served during forward playback
    const serveFrame = (chunkIndex) => {
        if (decodeForward && decodedBlocksCacheSize > 1) {
            for (let i = 1; i <= 3; i++) ensureChunkFetched(chunkIndex + i);
            if (!frameDataCache[jobID].activeChunkRequest) {
                startNextChunkDecode(chunkIndex + 1);
            }
        }
    };

    return { cache, serveFrame, frameDataCache, jobID };
}

// ---------------------------------------------------------------------------
// Test 1: Parallel network fetches — serving a frame starts 3 ahead
// ---------------------------------------------------------------------------
console.log('\nTest 1: Parallel network fetches');
{
    const provider = makeProvider();
    provider.markCached(0); // chunk 0 already decoded (first load done)
    const { cache, serveFrame } = makeCache(provider);

    serveFrame(0); // serve a frame from chunk 0

    assert(cache.fetchCalls.length === 3, `3 fetch calls started (got ${cache.fetchCalls.length})`);
    assert(cache.fetchCalls.includes(1), 'chunk 1 fetch started');
    assert(cache.fetchCalls.includes(2), 'chunk 2 fetch started');
    assert(cache.fetchCalls.includes(3), 'chunk 3 fetch started');
    assert(cache.rawChunkFetchCache.size === 3, 'rawChunkFetchCache has 3 entries');
}

// ---------------------------------------------------------------------------
// Test 2: ensureChunkFetched is idempotent — no double fetches
// ---------------------------------------------------------------------------
console.log('\nTest 2: idempotent ensureChunkFetched');
{
    const provider = makeProvider();
    provider.markCached(0);
    const { cache, serveFrame } = makeCache(provider);

    serveFrame(0); // first frame — fetches 1, 2, 3
    serveFrame(0); // second frame in same chunk — should not re-fetch
    serveFrame(0); // third frame — same

    assert(cache.fetchCalls.length === 3, `exactly 3 fetches across multiple frames (got ${cache.fetchCalls.length})`);
}

// ---------------------------------------------------------------------------
// Test 3: Single decode gate — only 1 decode at a time
// ---------------------------------------------------------------------------
console.log('\nTest 3: Single decode gate');
{
    const provider = makeProvider();
    provider.markCached(0);
    const { cache, serveFrame } = makeCache(provider);

    serveFrame(0); // triggers startNextChunkDecode(1)
    assert(cache.activeChunkRequest !== null, 'activeChunkRequest set after first frame');

    serveFrame(0); // activeChunkRequest is set, should NOT trigger another decode
    assert(provider.decodeCalls.length === 0, 'no decode call yet (fetch still pending)');

    // let fetch+decode microtasks run
    await new Promise(r => setTimeout(r, 20));
    assert(provider.decodeCalls.length === 1, `exactly 1 decode call made (got ${provider.decodeCalls.length})`);
    assert(provider.decodeCalls[0] === 1, 'decoded chunk 1');
}

// ---------------------------------------------------------------------------
// Test 4: activeChunkRequest cleared after decode, frame becomes cached
// ---------------------------------------------------------------------------
console.log('\nTest 4: activeChunkRequest cleared and chunk cached after decode');
{
    const provider = makeProvider();
    provider.markCached(0);
    const { cache, serveFrame } = makeCache(provider);

    serveFrame(0); // start prefetch + decode of chunk 1
    const req = cache.activeChunkRequest;
    assert(req !== null, 'activeChunkRequest set');

    await new Promise(r => setTimeout(r, 20)); // let microtasks settle

    assert(provider.isChunkCached(1), 'chunk 1 now cached after decode');
    assert(cache.activeChunkRequest === null, 'activeChunkRequest cleared after decode');
    assert(!cache.rawChunkFetchCache.has(1), 'rawChunkFetchCache entry for chunk 1 removed');
}

// ---------------------------------------------------------------------------
// Test 5: Frame at chunk boundary waits for prefetch and resolves correctly
// ---------------------------------------------------------------------------
console.log('\nTest 5: Chunk boundary frame uses prefetched data');
{
    const provider = makeProvider();
    provider.markCached(0);
    const { cache, serveFrame, frameDataCache, jobID } = makeCache(provider);

    // Simulate: serve several frames from chunk 0, triggering prefetch of chunk 1
    serveFrame(0);
    await new Promise(r => setTimeout(r, 5)); // fetch resolves, decode starts

    // Now simulate chunk boundary: frame in chunk 1 is not cached yet
    // The mandatory decode path waits on activeChunkRequest
    const pendingChunkRequest = frameDataCache[jobID].activeChunkRequest || Promise.resolve();
    let frameResolvedFromCache = false;
    pendingChunkRequest.finally(() => {
        if (provider.isChunkCached(1)) frameResolvedFromCache = true;
    });

    await new Promise(r => setTimeout(r, 30)); // let decode complete

    assert(frameResolvedFromCache, 'frame at chunk boundary resolved from prefetched cache');
}

// ---------------------------------------------------------------------------
// Test 6: No eviction of current chunk — only 1 chunk decoded ahead
// ---------------------------------------------------------------------------
console.log('\nTest 6: No eviction of current chunk');
{
    const provider = makeProvider();
    provider.markCached(0);
    const { cache, serveFrame } = makeCache(provider);

    // Even serving many frames from chunk 0, only 1 decode should happen
    for (let i = 0; i < 36; i++) serveFrame(0);

    await new Promise(r => setTimeout(r, 50));

    // Only chunk 1 should be decoded — NOT 2 or 3 (no auto-chaining)
    const decoded = provider.decodeCalls;
    assert(decoded.length === 1, `only 1 chunk decoded ahead, got: [${decoded}]`);
    assert(decoded[0] === 1, 'that chunk is chunk 1 (immediate next)');
    assert(provider.isChunkCached(0), 'chunk 0 still cached (not evicted)');
    assert(!provider.isChunkCached(2), 'chunk 2 NOT pre-decoded (no runaway chaining)');
    assert(!provider.isChunkCached(3), 'chunk 3 NOT pre-decoded');
}

// ---------------------------------------------------------------------------
// Test 7: Already-cached chunks are skipped in fetch loop
// ---------------------------------------------------------------------------
console.log('\nTest 7: Already-cached chunks skipped in ensureChunkFetched');
{
    const provider = makeProvider();
    provider.markCached(0);
    provider.markCached(1); // chunk 1 already cached
    const { cache, serveFrame } = makeCache(provider);

    // The window is fixed at N+1, N+2, N+3. Chunk 1 is cached so only 2 and 3 are fetched.
    // The window does NOT slide to compensate for cached entries (intentional — keep it simple).
    serveFrame(0);

    assert(!cache.fetchCalls.includes(1), 'no re-fetch of already-cached chunk 1');
    assert(cache.fetchCalls.includes(2), 'chunk 2 fetched');
    assert(cache.fetchCalls.includes(3), 'chunk 3 fetched');
    assert(!cache.fetchCalls.includes(4), 'chunk 4 not fetched (window is fixed, not sliding)');
    assert(cache.fetchCalls.length === 2, `2 fetches for un-cached chunks in window (got ${cache.fetchCalls.length})`);
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------
console.log(`\n${'─'.repeat(50)}`);
console.log(`Results: ${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
