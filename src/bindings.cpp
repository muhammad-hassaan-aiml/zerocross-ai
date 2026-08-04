#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstring>

#include "game_state.h"
#include "mcts.h"

namespace py = pybind11;

PYBIND11_MODULE(zerocross_engine, m) {
    m.doc() = "ZeroCross Ultimate Tic-Tac-Toe C++ Engine";

    py::class_<GameState>(m, "GameState")
        .def(py::init<>())
        .def("encode", &GameState::encode)
        .def("legal_mask", &GameState::legal_mask)
        .def("play", &GameState::play)
        .def("is_terminal", &GameState::is_terminal)
        .def("get_current_player", &GameState::get_current_player) 
        .def("get_winner", &GameState::get_winner)                 
        .def_static("from_array", &GameState::from_array);

    py::class_<MCTSTree>(m, "MCTSTree")
        .def(py::init<const GameState&, bool>(), py::arg("root_state"), py::arg("add_noise") = true)
        .def("request_leaves", [](MCTSTree& self, int batch_size) {
            auto leaves = self.request_leaves(batch_size);
            size_t num_leaves = leaves.size();
            
            py::array_t<float> result({num_leaves, (size_t)6, (size_t)9, (size_t)9});
            auto buf = result.request();
            float* ptr = static_cast<float*>(buf.ptr);
            
            for (size_t i = 0; i < num_leaves; ++i) {
                std::memcpy(ptr + i * 486, leaves[i].data(), 486 * sizeof(float));
            }
            return result;
        })
        .def("submit_results", [](MCTSTree& self, py::array_t<float> policies, py::array_t<float> values) {
            auto buf_p = policies.request();
            auto buf_v = values.request();
            size_t num_leaves = buf_p.shape[0];
            
            std::vector<std::vector<float>> pol_vec(num_leaves, std::vector<float>(81));
            std::vector<float> val_vec(num_leaves);
            
            float* p_ptr = static_cast<float*>(buf_p.ptr);
            float* v_ptr = static_cast<float*>(buf_v.ptr);
            
            for (size_t i = 0; i < num_leaves; ++i) {
                std::memcpy(pol_vec[i].data(), p_ptr + i * 81, 81 * sizeof(float));
                val_vec[i] = v_ptr[i];
            }
            self.submit_results(pol_vec, val_vec);
        })
        .def("is_done", &MCTSTree::is_done)
        .def("root_policy", &MCTSTree::root_policy)
        .def("advance", &MCTSTree::advance);

    py::class_<ParallelMCTS>(m, "ParallelMCTS")
        .def(py::init<int, bool>(), py::arg("num_games"), py::arg("add_noise") = true)
        .def("set_state", &ParallelMCTS::set_state)
        .def("advance", &ParallelMCTS::advance)
        .def("is_done", &ParallelMCTS::is_done)
        .def("root_policy", &ParallelMCTS::root_policy)
        .def("request_batch", [](ParallelMCTS& self, int n_simulations, int batch_per_tree) {
            auto leaves = self.request_batch(n_simulations, batch_per_tree);
            size_t num_leaves = leaves.size();
            
            py::array_t<float> result_leaves({num_leaves, (size_t)6, (size_t)9, (size_t)9});
            auto buf_leaves = result_leaves.request();
            float* ptr_leaves = static_cast<float*>(buf_leaves.ptr);
            
            for (size_t i = 0; i < num_leaves; ++i) {
                std::memcpy(ptr_leaves + i * 486, leaves[i].data(), 486 * sizeof(float));
            }

            const auto& mapping = self.get_leaf_game_mapping();
            py::array_t<int> result_mapping(mapping.size());
            auto buf_mapping = result_mapping.request();
            int* ptr_mapping = static_cast<int*>(buf_mapping.ptr);
            std::memcpy(ptr_mapping, mapping.data(), mapping.size() * sizeof(int));
            
            return py::make_tuple(result_leaves, result_mapping);
        })
        .def("submit_batch", [](ParallelMCTS& self, py::array_t<float> policies, py::array_t<float> values) {
            auto buf_p = policies.request();
            auto buf_v = values.request();
            size_t num_leaves = buf_p.shape[0];
            
            std::vector<std::vector<float>> pol_vec(num_leaves, std::vector<float>(81));
            std::vector<float> val_vec(num_leaves);
            
            float* p_ptr = static_cast<float*>(buf_p.ptr);
            float* v_ptr = static_cast<float*>(buf_v.ptr);
            
            for (size_t i = 0; i < num_leaves; ++i) {
                std::memcpy(pol_vec[i].data(), p_ptr + i * 81, 81 * sizeof(float));
                val_vec[i] = v_ptr[i];
            }
            self.submit_batch(pol_vec, val_vec);
        });
}