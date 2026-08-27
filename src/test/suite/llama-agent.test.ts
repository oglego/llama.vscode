/// <reference types="mocha" />
import * as assert from 'assert';
import { suite, test } from 'mocha';
import { LlamaAgent } from '../../llama-agent';

suite('LlamaAgent Test Suite', () => {
	test('caps accumulated reasoning text', () => {
		const agent = Object.create(LlamaAgent.prototype) as any;

		agent.reasoningText = '';
		agent.appendReasoning('x'.repeat(25000));

		assert.ok(agent.reasoningText.startsWith('[reasoning truncated]\n'));
		assert.ok(agent.reasoningText.length <= 20000);
	});

	test('accumulates reasoning across tool-loop iterations with a separator', () => {
		const agent = Object.create(LlamaAgent.prototype) as any;

		// Simulate two iterations of the tool-calling loop: each iteration
		// inserts the separator (when reasoning already exists) before the
		// next chunk of reasoning streams in, mirroring what runAgent does.
		agent.reasoningText = '';
		agent.appendReasoning('step one reasoning');

		const separator = (LlamaAgent as any).REASONING_ITERATION_SEPARATOR as string;
		if (agent.reasoningText) {
			agent.reasoningText += separator;
		}
		agent.appendReasoning('step two reasoning');

		assert.strictEqual(
			agent.reasoningText,
			'step one reasoning' + separator + 'step two reasoning'
		);
	});
});
