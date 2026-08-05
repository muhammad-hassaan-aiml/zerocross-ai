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
            // Extract the unique gameId
            const { gameId, board, activeGrid, simulations } = data;
            
            const state = wasmModule.create_state(board, activeGrid);
            const tree = new wasmModule.MCTSTree(state, false);

            // Determine the optimal batch size based on the difficulty.
            // This prevents MCTS "tunnel vision" by ensuring the tree updates
            // frequently enough, while still utilizing CPU SIMD speed optimizations.
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

            const policyView = wasmModule.get_root_policy(tree, 0.0);
            const policyArray = new Float32Array(policyView);

            let ai_move = -1;
            let maxP = -Infinity;
            for (let i = 0; i < 81; i++) {
                if (policyArray[i] > maxP) {
                    maxP = policyArray[i];
                    ai_move = i;
                }
            }

            const rootStateView = wasmModule.encode_state(state);
            const rootTensorData = new Float32Array(rootStateView);
            const rootTensor = new ort.Tensor('float32', rootTensorData, [1, 6, 9, 9]);
            const rootResults = await ortSession.run({ 'input': rootTensor });
            const rootVal = rootResults['value'].data[0];

            tree.delete();
            state.delete();

            // Echo the gameId back to the main thread
            self.postMessage({ type: 'think_done', gameId: gameId, move: ai_move, rootVal: rootVal });

        } catch (error) {
            self.postMessage({ type: 'think_error', error: error.message || error });
        }
    }
};