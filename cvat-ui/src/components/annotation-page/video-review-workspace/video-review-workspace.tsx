// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, {
    useState, useEffect, useCallback, useRef,
} from 'react';
import { useSelector, useDispatch } from 'react-redux';
import Layout from 'antd/lib/layout';
import InputNumber from 'antd/lib/input-number';
import Button from 'antd/lib/button';
import notification from 'antd/lib/notification';

import { CombinedState } from 'reducers';
import { createAnnotationsAsync, fetchAnnotationsAsync } from 'actions/annotation-actions';
import {
    ObjectState, ObjectType, Label, LabelType,
} from 'cvat-core-wrapper';

const LABEL_NAME = 'Video Review';
const ATTR_NAME = 'object_count';

async function ensureLabel(jobInstance: any): Promise<Label> {
    const task = jobInstance.task || await jobInstance.task;
    const projectId = task?.projectId ?? jobInstance.projectId;

    // Check existing labels on the job
    const existing = jobInstance.labels.find(
        (l: Label) => l.name === LABEL_NAME && l.type === LabelType.TAG,
    );
    if (existing) return existing;

    // Create the label on whichever entity owns labels
    const labelData = {
        name: LABEL_NAME,
        type: LabelType.TAG,
        color: '#aa55ff',
        attributes: [{
            name: ATTR_NAME,
            mutable: false,
            input_type: 'number',
            default_value: '0',
            values: ['0', '1000000'],
        }],
    };

    if (projectId) {
        const [project] = await (window as any).cvat.projects.get({ id: projectId });
        project.labels = [...project.labels, new Label(labelData)];
        await project.save();
        // Refresh job to pick up the new label
        const [refreshed] = await (window as any).cvat.jobs.get({ jobID: jobInstance.id });
        return refreshed.labels.find((l: Label) => l.name === LABEL_NAME);
    }

    // Task-level label
    task.labels = [...task.labels, new Label(labelData)];
    await task.save();
    const [refreshed] = await (window as any).cvat.jobs.get({ jobID: jobInstance.id });
    return refreshed.labels.find((l: Label) => l.name === LABEL_NAME);
}

function VideoReviewWorkspace(): JSX.Element {
    const dispatch = useDispatch();
    const jobInstance = useSelector((state: CombinedState) => state.annotation.job.instance);
    const frameNumber = useSelector((state: CombinedState) => state.annotation.player.frame.number);
    const states: ObjectState[] = useSelector(
        (state: CombinedState) => state.annotation.annotations.states,
    );

    const [count, setCount] = useState<number>(0);
    const [saving, setSaving] = useState(false);
    const [label, setLabel] = useState<Label | null>(null);
    const mountedRef = useRef(true);

    // Ensure the "Video Review" label exists
    useEffect(() => {
        if (!jobInstance) return;
        ensureLabel(jobInstance).then((l) => {
            if (mountedRef.current && l) setLabel(l);
        }).catch((err) => {
            notification.error({
                message: 'Video Review',
                description: `Failed to create label: ${err.message}`,
            });
        });
    }, [jobInstance]);

    // Load existing count from annotations when frame changes
    useEffect(() => {
        if (!label) return;
        const existing = states.find(
            (s: ObjectState) => s.objectType === ObjectType.TAG &&
                s.label.id === label.id &&
                s.frame === frameNumber,
        );
        if (existing) {
            const attr = label.attributes.find((a: any) => a.name === ATTR_NAME);
            if (attr && existing.attributes[attr.id] !== undefined) {
                setCount(Number(existing.attributes[attr.id]) || 0);
                return;
            }
        }
        setCount(0);
    }, [frameNumber, states, label]);

    // Capture spacebar to prevent playback toggle
    useEffect(() => {
        const handler = (e: KeyboardEvent): void => {
            if (e.code === 'Space') {
                e.stopPropagation();
                e.preventDefault();
            }
        };
        window.addEventListener('keydown', handler, true);
        return () => {
            mountedRef.current = false;
            window.removeEventListener('keydown', handler, true);
        };
    }, []);

    const save = useCallback(async () => {
        if (!jobInstance || !label) return;
        setSaving(true);
        try {
            const attr = label.attributes.find((a: any) => a.name === ATTR_NAME);
            if (!attr) throw new Error('object_count attribute not found');

            const state = new ObjectState({
                objectType: ObjectType.TAG,
                label,
                frame: frameNumber,
                attributes: { [attr.id]: String(count) },
            });

            await dispatch(createAnnotationsAsync([state]) as any);
            notification.success({
                message: 'Video Review',
                description: `Saved count: ${count} for frame ${frameNumber}`,
            });
            dispatch(fetchAnnotationsAsync() as any);
        } catch (error: unknown) {
            notification.error({
                message: 'Video Review',
                description: error instanceof Error ? error.message : 'Save failed',
            });
        } finally {
            setSaving(false);
        }
    }, [jobInstance, label, frameNumber, count, dispatch]);

    return (
        <Layout className='cvat-video-review-workspace'>
            <Layout.Content
                style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    height: '100%',
                    gap: 24,
                    padding: 48,
                }}
            >
                <h2 style={{ margin: 0 }}>
                    Frame
                    {' '}
                    {frameNumber}
                </h2>

                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <span style={{ fontWeight: 500 }}>Object Count:</span>
                    <InputNumber
                        min={0}
                        value={count}
                        onChange={(val) => { if (val !== null) setCount(val); }}
                        style={{ width: 120 }}
                        size='large'
                    />
                </div>

                <Button
                    type='primary'
                    size='large'
                    onClick={save}
                    loading={saving}
                    disabled={!label}
                >
                    Save
                </Button>
            </Layout.Content>
        </Layout>
    );
}

export default React.memo(VideoReviewWorkspace);
