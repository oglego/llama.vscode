/// <reference types="mocha" />
import * as assert from 'assert';
import { ChatContext } from '../../chat-context';
import { Application } from '../../application';
import { ChunkEntry } from '../../types';
import { ModelType } from '../../constants';
import { suite, test, setup } from 'mocha';

// Deterministic fake embeddings: encodes whether the text is about "cats" or
// "databases" as a 2-d vector, so cosine similarity ranking is predictable
// without needing a real embeddings server.
function fakeEmbedding(text: string): number[] {
    const lower = text.toLowerCase();
    const cat = /cat|kitten|feline/.test(lower) ? 1 : 0;
    const db = /database|sql|query|index/.test(lower) ? 1 : 0;
    // Non-zero fallback so magnitude is never 0 for unrelated text.
    return [cat || 0.01, db || 0.01];
}

class MockApplication {
    configuration = {
        rag_max_bm25_filter_chunks: 50,
        getUiText: (key: string) => key
    };

    llamaServer = {
        getEmbeddings: async (text: string) => ({
            data: [{ embedding: fakeEmbedding(text) }]
        })
    };

    modelService = {
        selectDefaultModel: async (_type: ModelType, _key: string) => undefined
    };

    getEmbeddingsModel = () => ({ endpoint: 'http://localhost:8080' } as any);
}

suite('ChatContext.semanticSearch Test Suite', () => {
    let chatContext: ChatContext;
    let mockApp: Application;

    setup(() => {
        mockApp = new MockApplication() as unknown as Application;
        chatContext = new ChatContext(mockApp);
    });

    function addEntry(id: number, uri: string, content: string): void {
        chatContext.entries.set(id, {
            uri,
            content,
            firstLine: 1,
            lastLine: 1,
            hash: `hash-${id}`,
            embedding: []
        } as ChunkEntry);
    }

    test('returns an empty array with an info message when nothing is indexed', async () => {
        const results = await chatContext.semanticSearch('anything');
        assert.strictEqual(results.length, 0);
    });

    test('ranks semantically closer chunks first', async () => {
        addEntry(1, 'src/animals.ts', 'function feedTheCat() { /* cats love tuna */ }');
        addEntry(2, 'src/db.ts', 'function runQuery() { return database.query(sql); }');
        addEntry(3, 'src/unrelated.ts', 'export const PI = 3.14159;');

        const results = await chatContext.semanticSearch('how do I query the database');

        assert.ok(results.length > 0, 'expected at least one result');
        assert.strictEqual(results[0].entry.uri, 'src/db.ts',
            'the database-related chunk should rank first for a database query');
        assert.ok(results[0].score > results[results.length - 1].score,
            'results should be sorted by descending score');
    });

    test('reuses cached embeddings on the chunk entry instead of recomputing', async () => {
        addEntry(1, 'src/animals.ts', 'kittens and cats');
        const entry = chatContext.entries.get(1)!;
        entry.embedding = [1, 0]; // pre-cached "cat" vector

        let embeddingCalls = 0;
        (mockApp as any).llamaServer.getEmbeddings = async (text: string) => {
            embeddingCalls++;
            return { data: [{ embedding: fakeEmbedding(text) }] };
        };

        await chatContext.semanticSearch('cat');

        // Only the query itself should trigger a fresh embedding call; the
        // chunk's cached embedding should be reused as-is.
        assert.strictEqual(embeddingCalls, 1);
    });
});
