importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');
importScripts('zerocross.js');

let wasmModule = null;
let ortSession = null;
let isReady = false;

async function initEngine() {
    try {
        wasmModule = await createZeroCrossModule();

        ort.env.wasm.numThreads = Math.max(1, (navigator.hardwareConcurrency || 4) - 1);
        ort.env.wasm.simd = true;
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

        const sessionOptions = {
            executionProviders: ['wasm']
        };

        ortSession = await ort.InferenceSession.create('best_model.onnx', sessionOptions);
        
        isReady = true;
        self.postMessage({ type: 'init_done' });
    } catch (error) {
        self.postMessage({ type: 'init_error', error: error.message || error });
    }
}

self.onmessage = async function(e) {
    const data = e.data;

    if (data.type === 'init') {
        initEngine();
    } 
    else if (data.type === 'think') {
        if (!isReady) return;

        try {
            const { gameId, board, activeGrid, simulations } = data;
            
            const state = wasmModule.create_state(board, activeGrid);
            const tree = new wasmModule.MCTSTree(state, false);

            let BATCH_SIZE = 8;
            if (simulations >= 800) {
                BATCH_SIZE = 32;
            } else if (simulations >= 200) {
                BATCH_SIZE = 16;
            }

            while (!tree.is_done(simulations)) {
                const leavesView = wasmModule.request_leaves(tree, BATCH_SIZE);
                const numLeaves = wasmModule.get_leaves_count();

                if (numLeaves > 0) {
                    const tensorData = new Float32Array(leavesView);
                    const tensor = new ort.Tensor('float32', tensorData, [numLeaves, 6, 9, 9]);

                    const results = await ortSession.run({ 'input': tensor });
                    const policies = results['policy_logits'].data;
                    const values = results['value'].data;

                    wasmModule.submit_results(tree, policies, values, numLeaves);
                }
            }

            // CHANGED: Pass 1.0 to get proportional visit counts instead of 0.0 (one-hot)
            const policyView = wasmModule.get_root_policy(tree, 1.0);
            const policyArray = new Float32Array(policyView);

            // Extract all legal moves and their probabilities
            let candidates = [];
            for (let i = 0; i < 81; i++) {
                if (policyArray[i] > 0) {
                    candidates.push({ move: i, prob: policyArray[i] });
                }
            }

            // Sort descending to find the best moves
            candidates.sort((a, b) => b.prob - a.prob);
            
            // Get the top 3
            const topCandidates = candidates.slice(0, 3);
            const ai_move = topCandidates.length > 0 ? topCandidates[0].move : -1;

            const rootStateView = wasmModule.encode_state(state);
            const rootTensorData = new Float32Array(rootStateView);
            const rootTensor = new ort.Tensor('float32', rootTensorData, [1, 6, 9, 9]);
            const rootResults = await ortSession.run({ 'input': rootTensor });
            const rootVal = rootResults['value'].data[0];

            tree.delete();
            state.delete();

            // Pass the candidates array back to the UI
            self.postMessage({ 
                type: 'think_done', 
                gameId: gameId, 
                move: ai_move, 
                rootVal: rootVal,
                candidates: topCandidates
            });

        } catch (error) {
            self.postMessage({ type: 'think_error', error: error.message || error });
        }
    }
};