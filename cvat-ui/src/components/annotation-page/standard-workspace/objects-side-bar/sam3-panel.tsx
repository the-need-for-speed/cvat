// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useState, useCallback } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import Button from 'antd/lib/button';
import Select from 'antd/lib/select';
import InputNumber from 'antd/lib/input-number';
import Input from 'antd/lib/input';
import notification from 'antd/lib/notification';
import Spin from 'antd/lib/spin';

import { CombinedState } from 'reducers';
import { createAnnotationsAsync } from 'actions/annotation-actions';
import { ObjectState, ObjectType, ShapeType, Label } from 'cvat-core-wrapper';

function SAM3Panel(): JSX.Element {
    const dispatch = useDispatch();

    const jobInstance = useSelector((state: CombinedState) => state.annotation.job.instance);
    const frameNumber = useSelector((state: CombinedState) => state.annotation.player.frame.number);
    const labels: Label[] = useSelector((state: CombinedState) => state.annotation.job.labels);

    const [selectedLabelID, setSelectedLabelID] = useState<number | null>(
        labels.length ? labels[0].id : null,
    );
    const [outputType, setOutputType] = useState<'polygon' | 'rectangle'>('polygon');
    const [confidence, setConfidence] = useState<number>(0.5);
    const [prompt, setPrompt] = useState<string>('');
    const [loading, setLoading] = useState(false);

    const run = useCallback(async () => {
        if (!jobInstance || selectedLabelID === null) return;

        const label = labels.find((l: Label) => l.id === selectedLabelID);
        if (!label) return;

        const textPrompt = prompt.trim() || label.name;

        setLoading(true);
        try {
            // Fetch the current frame as a blob
            const frameResp = await fetch(
                `/api/jobs/${jobInstance.id}/data?type=frame&quality=original&number=${frameNumber}`,
                { credentials: 'same-origin' },
            );
            if (!frameResp.ok) throw new Error(`Failed to fetch frame: ${frameResp.status}`);
            const blob = await frameResp.blob();

            // Convert to base64
            const arrayBuf = await blob.arrayBuffer();
            const b64 = btoa(
                new Uint8Array(arrayBuf).reduce((data, byte) => data + String.fromCharCode(byte), ''),
            );

            // Call SAM3 server
            const sam3Resp = await fetch('/sam3/segment_cvat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: b64,
                    prompt: textPrompt,
                    confidence,
                }),
            });
            if (!sam3Resp.ok) throw new Error(`SAM3 request failed: ${sam3Resp.status}`);

            const results: Array<{ points: number[]; confidence: number }> = await sam3Resp.json();

            if (!results.length) {
                notification.info({ message: 'SAM3', description: 'No objects detected.' });
                return;
            }

            const states = results.map((r) => {
                const shapeType = outputType === 'rectangle' ? ShapeType.RECTANGLE : ShapeType.POLYGON;
                let { points } = r;

                // For rectangles, convert polygon points to xtl,ytl,xbr,ybr
                if (shapeType === ShapeType.RECTANGLE && points.length > 4) {
                    const xs = points.filter((_, i) => i % 2 === 0);
                    const ys = points.filter((_, i) => i % 2 !== 0);
                    points = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
                }

                return new ObjectState({
                    objectType: ObjectType.SHAPE,
                    shapeType,
                    label,
                    frame: frameNumber,
                    points,
                    occluded: false,
                    zOrder: 0,
                });
            });

            dispatch(createAnnotationsAsync(states) as any);
            notification.success({
                message: 'SAM3',
                description: `Created ${states.length} annotation(s).`,
            });
        } catch (error: unknown) {
            notification.error({
                message: 'SAM3 Error',
                description: error instanceof Error ? error.message : 'Unknown error',
            });
        } finally {
            setLoading(false);
        }
    }, [jobInstance, frameNumber, labels, selectedLabelID, outputType, confidence, prompt, dispatch]);

    return (
        <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div>
                <span style={{ display: 'block', marginBottom: 4, fontWeight: 500 }}>Label</span>
                <Select
                    style={{ width: '100%' }}
                    value={selectedLabelID}
                    onChange={(val: number) => setSelectedLabelID(val)}
                >
                    {labels.map((l: Label) => (
                        <Select.Option key={l.id} value={l.id}>{l.name}</Select.Option>
                    ))}
                </Select>
            </div>

            <div>
                <span style={{ display: 'block', marginBottom: 4, fontWeight: 500 }}>
                    Text Prompt (blank = label name)
                </span>
                <Input
                    placeholder='e.g. person, car, dog...'
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                />
            </div>

            <div>
                <span style={{ display: 'block', marginBottom: 4, fontWeight: 500 }}>Output Type</span>
                <Select
                    style={{ width: '100%' }}
                    value={outputType}
                    onChange={(val: 'polygon' | 'rectangle') => setOutputType(val)}
                >
                    <Select.Option value='polygon'>Polygon</Select.Option>
                    <Select.Option value='rectangle'>Bounding Box</Select.Option>
                </Select>
            </div>

            <div>
                <span style={{ display: 'block', marginBottom: 4, fontWeight: 500 }}>Confidence</span>
                <InputNumber
                    style={{ width: '100%' }}
                    min={0}
                    max={1}
                    step={0.05}
                    value={confidence}
                    onChange={(val) => { if (val !== null) setConfidence(val); }}
                />
            </div>

            <Button
                type='primary'
                onClick={run}
                disabled={loading || selectedLabelID === null}
                block
            >
                {loading ? <Spin size='small' /> : 'Run SAM3'}
            </Button>
        </div>
    );
}

export default React.memo(SAM3Panel);
