#include <iostream>
#include "gumbel_cnode.h"

namespace tree{

    CSearchResults::CSearchResults(){
        this->num = 0;
    }

    CSearchResults::CSearchResults(int num){
        this->num = num;
        for(int i = 0; i < num; ++i){
            this->search_paths.push_back(std::vector<CNode*>());
            this->search_path_index_x_lst.push_back(std::vector<int>());
            this->search_path_index_y_lst.push_back(std::vector<int>());
            this->search_path_actions.push_back(std::vector<int>());
        }
    }

    CSearchResults::~CSearchResults(){}

    //*********************************************************

    CNode::CNode(){
        this->prior = 0;
        this->action_num = 0;
        this->best_action = -1;

        this->is_reset = 0;
        this->visit_count = 0;
        this->value_sum = 0;
        this->to_play = 0;
        this->reward_sum = 0.0;
        this->ptr_node_pool = nullptr;
        this->phase_added_flag = 0;
        this->current_phase = 0;
        this->phase_num = 0;
        this->phase_to_visit_num = 0;
        this->m = 0;
        this->value_mix = 0;
    }

    CNode::CNode(float prior, int action_num, std::vector<CNode>* ptr_node_pool){
        this->prior = prior;
        this->action_num = action_num;

        this->is_reset = 0;
        this->visit_count = 0;
        this->value_sum = 0;
        this->best_action = -1;
        this->to_play = 0;
        this->reward_sum = 0.0;
        this->ptr_node_pool = ptr_node_pool;
        this->hidden_state_index_x = -1;
        this->hidden_state_index_y = -1;
        this->is_root = 0;
        this->value_mix = 0;

        this->phase_added_flag = 0;
        this->current_phase = 0;
        this->phase_num = 0;
        this->phase_to_visit_num = 0;
        this->m = 0;
    }

    CNode::~CNode(){}

    void CNode::expand(int to_play, int hidden_state_index_x, int hidden_state_index_y, float reward_sum, const std::vector<float> &policy_logits, int simulation_num, int leaf_num){
        this->to_play = to_play;
        this->simulation_num = simulation_num;
        this->hidden_state_index_x = hidden_state_index_x;
        this->hidden_state_index_y = hidden_state_index_y;
        this->reward_sum = reward_sum;

        int action_num = this->action_num;

        float prior;
        for(int a = 0; a < action_num; ++a){
            int index = ptr_node_pool->size();
            this->children_index.push_back(index);
            this->ptr_node_pool->push_back(CNode(policy_logits[a], leaf_num, ptr_node_pool));
            this->ptr_node_pool->back().parent = this;
            this->prev_half_Qs.push_back(0.0);
        }

        if (DEBUG_MODE) {
            printf("expand prior: [");
            for(int a = 0; a < action_num; ++a){
                prior = this->get_child(a)->prior;
                printf("%f, ", prior);
            }
            printf("]\n");
        }
    }

    void CNode::print_out(){
        printf("*****\n");
        printf("visit count: %d \t hidden_state_index_x: %d \t hidden_state_index_y: %d \t reward: %f \t prior: %f \n.",
            this->visit_count, this->hidden_state_index_x, this->hidden_state_index_y, this->reward_sum, this->prior
        );
        printf("children_index size: %d \t pool size: %d \n.", this->children_index.size(), this->ptr_node_pool->size());
        printf("*****\n");
    }

    int CNode::expanded(){
        int child_num = this->children_index.size();
        if(child_num > 0) return 1;
        return 0;
    }

    float CNode::value(){
        float true_value = 0.0;
        if (this->visit_count == 0) {
            printf("%f\n", this->parent->value_mix);
            return this->parent->value_mix;
        }
        true_value = this->value_sum / this->visit_count;
        return true_value;
    }


    float CNode::get_qsa(int action, float discount){
        CNode* child = this->get_child(action);
        float true_reward = child->reward_sum - this->reward_sum;
        if (this->is_reset) {
            true_reward = child->reward_sum;
        }
        float qsa = true_reward + discount * child->value();
        return qsa;
    }

    float CNode::v_mix(float discount){
        float pi_dot_sum = 0.0;
        float pi_value_sum = 0.0;
        std::vector<float> pi_probs;
        for(int a = 0; a< this->action_num; ++a){
            CNode* child = this->get_child(a);
            pi_probs.push_back(exp(child->prior));
            pi_dot_sum += pi_probs.back();
        }
        float pi_visited_sum = 0.0;
        for(int a = 0; a < this->action_num; ++a) {
            pi_probs[a] = pi_probs[a] / pi_dot_sum;
            CNode* child = this->get_child(a);
            if(child->expanded()) {
                pi_visited_sum += pi_probs[a];
                pi_value_sum += pi_probs[a] * this->get_qsa(a, discount);
            }
        }
        if (abs(pi_visited_sum - 0.0) < 0.0001) {
            return this->value();
        }
        return (1. / (1. + this->visit_count)) * (this->value() + (this->visit_count / pi_visited_sum) * pi_value_sum);
    }

    std::vector<float> CNode::completedQ(float discount){
        std::vector<float> completedQ;
        float v_mix = this->v_mix(discount);
        this->value_mix = v_mix;
        for (int a = 0; a < this->action_num; ++a){
            CNode* child = this->get_child(a);
            if (child->expanded()) {
                completedQ.push_back(this->get_qsa(a, discount));
            } else {
                completedQ.push_back(v_mix);
            }
        }
        return completedQ;
    }

    std::vector<int> CNode::get_trajectory(){
        std::vector<int> traj;

        CNode* node = this;
        int best_action = node->best_action;
        while(best_action >= 0){
            traj.push_back(best_action);

            node = node->get_child(best_action);
            best_action = node->best_action;
        }
        return traj;
    }

    CNode* CNode::get_child(int action){
        int index = this->children_index[action];
        return &((*(this->ptr_node_pool))[index]);
    }

    //*********************************************************

    CRoots::CRoots(){
        this->root_num = 0;
        this->action_num = 0;
        this->pool_size = 0;
    }

    CRoots::CRoots(int root_num, int action_num, int pool_size){
        this->root_num = root_num;
        this->action_num = action_num;
        this->pool_size = pool_size;

        this->node_pools.reserve(root_num);
        this->roots.reserve(root_num);

        for(int i = 0; i < root_num; ++i){
            this->node_pools.push_back(std::vector<CNode>());
            this->node_pools[i].reserve(pool_size);

            this->roots.push_back(CNode(0, action_num, &this->node_pools[i]));
        }
    }

    CRoots::~CRoots(){}

    void CRoots::prepare(const std::vector<float> &reward_sums, const std::vector<std::vector<float>> &policies, int m, int simulation_num, const std::vector<float> &values, int leaf_num){
        for(int i = 0; i < this->root_num; ++i){
            this->roots[i].expand(0, 0, i, reward_sums[i], policies[i], simulation_num, leaf_num);
            this->roots[i].is_root = 1;
            this->roots[i].m = std::min(m, this->action_num);
            this->roots[i].phase_num = ceil(log2(this->roots[i].m));
            this->roots[i].simulation_num = simulation_num;
            this->roots[i].value_sum += values[i];
            this->roots[i].visit_count += 1;
        }

        if(DEBUG_MODE){
            for(int i = 0; i < this->root_num; ++i){
                printf("change prior with noise: [");
                for(int a = 0; a < action_num; ++a){
                    float prior = this->roots[i].get_child(a)->prior;
                    printf("%f, ", prior);
                }
                printf("]\n");
            }
        }
    }

    void CRoots::clear(){
        this->node_pools.clear();
        this->roots.clear();
    }

    std::vector<std::vector<int>> CRoots::get_trajectories(){
        std::vector<std::vector<int>> trajs;
        trajs.reserve(this->root_num);

        for(int i = 0; i < this->root_num; ++i){
            trajs.push_back(this->roots[i].get_trajectory());
        }
        return trajs;
    }

    std::vector<std::vector<float>> CRoots::get_advantages(float discount){
        std::vector<std::vector<float>> advantages;
        advantages.reserve(this->root_num);
        for(int i = 0; i < this->root_num; ++i){
            CNode* root = &(this->roots[i]);
            std::vector<float> advantage = calc_advantage(root, discount);
            advantages.push_back(advantage);
        }
        return advantages;
    }

    std::vector<std::vector<float>> CRoots::get_pi_primes(tools::CMinMaxStatsList *min_max_stats_lst, float c_visit, float c_scale, float discount){
        std::vector<std::vector<float>> pi_primes;
        pi_primes.reserve(this->root_num);

        for(int i = 0; i < this->root_num; ++i){
            CNode* root = &(this->roots[i]);
            std::vector<float> pi_prime = calc_pi_prime_dot(root, min_max_stats_lst->stats_lst[i], c_visit, c_scale, discount);
            pi_primes.push_back(pi_prime);
        }
        return pi_primes;
    }

    std::vector<std::vector<float>> CRoots::get_priors(){
        std::vector<std::vector<float>> roots_priors;
        std::vector<float> tmp;
        for (int i = 0; i < this->root_num; ++i){
            tmp.clear();
            for (int a = 0; a < this->roots[i].action_num; ++a){
                CNode* child = this->roots[i].get_child(a);
                tmp.push_back(child->prior);
            }
            roots_priors.push_back(tmp);
        }
        return roots_priors;
    }

    std::vector<int> CRoots::get_actions(tools::CMinMaxStatsList *min_max_stats_lst, float c_visit, float c_scale, const std::vector<std::vector<float>> &gumbels, float discount){
        std::vector<int> actions;
        for(int i = 0; i < this->root_num; ++i){
            CNode* root = &(this->roots[i]);
            std::vector<std::pair<int, float>> scores = calc_gumbel_score(root, gumbels[i], min_max_stats_lst->stats_lst[i], c_visit, c_scale, discount);
            std::vector<float> temp_scores;
            for (auto score : scores){
                temp_scores.push_back(score.second);
            }
            int action = argmax(temp_scores);
            action = root->selected_children[action].first;
            actions.push_back(action);
        }
        return actions;
    }

    std::vector<float> calc_advantage(CNode* node, float discount){
        std::vector<float> advantage;
        std::vector<float> completedQ = node->completedQ(discount);
        for (int a = 0; a < node->action_num; ++a){
//            CNode* child = node->get_child(a);
            advantage.push_back(completedQ[a] - node->value());     // target_V - this_V
        }
        return advantage;
    }

    std::vector<float> calc_pi_prime(CNode* node, tools::CMinMaxStats &min_max_stats, float c_visit, float c_scale, float discount, int final){
        std::vector<float> pi_prime;
        std::vector<float> completedQ = node->completedQ(discount);
        std::vector<float> sigmaQ;
        float pi_prime_max = -10000.0;
        for (int a = 0; a < node->action_num; ++a){
            CNode* child = node->get_child(a);
            float normalized_value = min_max_stats.normalize(completedQ[a]);
//            float visit_count = std::max(child->visit_count, 1);
            if (normalized_value < 0) normalized_value = 0;
            if (normalized_value > 1) normalized_value = 1;
            sigmaQ.push_back(sigma(normalized_value, node, c_visit, c_scale));
            float score = child->prior + sigmaQ[a];
//            float score = child->prior + sigma(normalized_value * std::sqrt(visit_count), node, c_visit, c_scale);
//            float score = child->prior + sigma(normalized_value * visit_count / node->simulation_num, node, c_visit, c_scale);
            pi_prime_max = std::max(pi_prime_max, score);
            pi_prime.push_back(score);
        }
        float pi_prime_sum = 0.0, pi_value_sum = 0.0;
        for (int a = 0; a < node->action_num; ++a){
            pi_prime[a] = exp(pi_prime[a] - pi_prime_max);
            pi_value_sum += pi_prime[a];
        }
        for (int a = 0; a < node->action_num; ++a){
            pi_prime[a] = pi_prime[a] / pi_value_sum;
        }

        return pi_prime;
    }

    std::vector<float> calc_pi_prime_dot(CNode* node, tools::CMinMaxStats &min_max_stats, float c_visit, float c_scale, float discount) {
        std::vector<float> pi_prime;
        std::vector<float> completedQ = node->completedQ(discount);
        float pi_prime_max = -10000.0;
        for (int a = 0; a < node->action_num; ++a){
            CNode* child = node->get_child(a);
            float normalized_value = min_max_stats.normalize(completedQ[a]);
            float visit_count = std::max(child->visit_count, 1);
            if (normalized_value < 0) normalized_value = 0;
            if (normalized_value > 1) normalized_value = 1;
            float normalized_prev_half_Q = min_max_stats.normalize(node->prev_half_Qs[a]);
            if (normalized_prev_half_Q < 0) normalized_prev_half_Q = 0;
            if (normalized_prev_half_Q > 1) normalized_prev_half_Q = 1;
            float ratio = 3.0;
//            float score = child->prior + sigma(normalized_value * std::sqrt(visit_count), node, c_visit, c_scale);
//            float score = child->prior + sigma(normalized_value * visit_count / node->simulation_num, node, c_visit, c_scale);
//            float score = child->prior + sigma(normalized_value * std::log(visit_count + 1), node, c_visit, c_scale);
            float score = child->prior + sigma(normalized_value, node, c_visit, c_scale);
//            printf("action=%d, n_value=%.4f, n_prev_half=%.4f, score=%.4f, penalty=%.4f\n", a, normalized_value, normalized_prev_half_Q, score, 10*std::abs(normalized_value - normalized_prev_half_Q));
//            score += ratio * (normalized_value - normalized_prev_half_Q);
            pi_prime_max = std::max(pi_prime_max, score);
            pi_prime.push_back(score);
        }
//        deactivate if using GRPO
//        float pi_prime_sum = 0.0, pi_value_sum = 0.0;
//        for (int a = 0; a < node->action_num; ++a){
//            pi_prime[a] = exp(pi_prime[a] - pi_prime_max);
//            pi_value_sum += pi_prime[a];
//        }
//        for (int a = 0; a < node->action_num; ++a){
//            pi_prime[a] = pi_prime[a] / pi_value_sum;
//        }
        return pi_prime;
    }

    std::vector<std::pair<int, float>> calc_gumbel_score(CNode* node, const std::vector<float> &gumbels, tools::CMinMaxStats &min_max_stats, float c_visit, float c_scale, float discount){
        std::vector<std::pair<int, float>> gumbel_scores;
        std::vector<float> completedQ = node->completedQ(discount);
        for (auto selected_child : node->selected_children){
            int a = selected_child.first;
//            float normalized_value = min_max_stats.normalize(selected_child.second->value(node));
            float normalized_value = min_max_stats.normalize(completedQ[a]);
            if (normalized_value < 0) normalized_value = 0;
            if (normalized_value > 1) normalized_value = 1;
            float normalized_prev_half_Q = min_max_stats.normalize(node->prev_half_Qs[a]);
            if (normalized_prev_half_Q < 0) normalized_prev_half_Q = 0;
            if (normalized_prev_half_Q > 1) normalized_prev_half_Q = 1;
            float ratio = 3.0;
            int visit_count = std::max(selected_child.second->visit_count, 1);
            float score = gumbels[a] + selected_child.second->prior + sigma(normalized_value, node, c_visit, c_scale);
//            score += ratio * (normalized_value - normalized_prev_half_Q);
            gumbel_scores.push_back(std::make_pair(a, score));
//            gumbel_scores.push_back(std::make_pair(a, gumbels[a] + selected_child.second->prior + sigma(normalized_value * std::sqrt(visit_count), node, c_visit, c_scale)));
//            gumbel_scores.push_back(std::make_pair(a, gumbels[a] + selected_child.second->prior + sigma(normalized_value * visit_count / node->simulation_num, node, c_visit, c_scale)));
        }
        return gumbel_scores;
    }

    std::vector<float> calc_non_root_score(CNode* node, tools::CMinMaxStats &min_max_stats, float c_visit, float c_scale, float discount){
        std::vector<float> pi_primes = calc_pi_prime(node, min_max_stats, c_visit, c_scale, discount, 0);
        std::vector<float> scores;
        for (int a = 0; a < node->action_num; ++a){
            CNode* child = node->get_child(a);
            scores.push_back(pi_primes[a] - float(child->visit_count / (1.0 + node->visit_count)));
        }
        return scores;
    }

    bool compare(std::pair<int, float> &a, std::pair<int, float> &b){
            return a.second > b.second;
        }

    bool compare_inv(std::pair<int, float> &a, std::pair<int, float> &b){
            return a.second < b.second;
        }

    void sequential_halving(CNode* root, int simulation_idx, tools::CMinMaxStats &min_max_stats, const std::vector<float> &gumbels, float c_visit, float c_scale, float discount){

        if(root->phase_added_flag == 0){
            if (root->current_phase < root->phase_num - 1) {
                root->phase_to_visit_num += int(std::max(1, int(float(root->simulation_num) / float(root->phase_num) / float(root->m)))) * root->m;
                root->phase_to_visit_num = std::min(root->phase_to_visit_num, root->simulation_num);
                root->phase_added_flag = 1;
            }
            else if (root->current_phase == root->phase_num - 1) {
                root->phase_to_visit_num = root->simulation_num;
                root->phase_added_flag = 1;
            }
        }

        if ((simulation_idx + 1) >= root->phase_to_visit_num) {
            if (root->selected_children.size() >= 2){
                int current_num = root->selected_children.size();
                std::vector<std::pair<int, float>> values = calc_gumbel_score(root, gumbels, min_max_stats, c_visit, c_scale, discount);
                if (root->current_phase == 0){
                    std::vector<float> completedQ = root->completedQ(discount);
                    for (auto selected_child: root->selected_children){
                        int a = selected_child.first;
                        root->prev_half_Qs[a] = completedQ[a];
//                        printf("%d, %.2f\n", a, completedQ[a]);
                    }
                }
                std::sort(values.begin(), values.end(), compare);
                root->selected_children.clear();
                for (int j = 0; j < int(values.size() / 2.0); ++j){
                    int a = values[j].first;
                    CNode* child = root->get_child(a);
                    root->selected_children.push_back(std::make_pair(a, child));
                }
                root->m = root->selected_children.size();
                root->current_phase += 1;
                root->phase_added_flag = 0;
            }
        }
    }

    std::vector<float> CRoots::get_values(){
        std::vector<float> values;
        for(int i = 0; i < this->root_num; ++i){
            CNode* root = &(this->roots[i]);
            values.push_back(this->roots[i].value());
        }
        return values;
    }

    std::vector<std::vector<float>> CRoots::get_child_values(float discount){
        std::vector<std::vector<float>> child_values;
        for(int i=0; i<this->root_num; ++i){
            CNode* root = &(this->roots[i]);
            child_values.push_back(root->completedQ(discount));
        }
        return child_values;
    }

    //*********************************************************

    void cback_propagate(std::vector<CNode*> &search_path, tools::CMinMaxStats &min_max_stats, int to_play, float value, float discount){
        float bootstrap_value = value;
        int path_len = search_path.size();
        for(int i = path_len - 1; i >= 0; --i){
            CNode* node = search_path[i];
            node->value_sum += bootstrap_value;
            node->visit_count += 1;

            float parent_reward_sum = 0.0;
            int is_reset = 0;
            if(i >= 1){
                CNode* parent = search_path[i - 1];
                parent_reward_sum = parent->reward_sum;
                is_reset = parent->is_reset;
            }

            float true_reward = node->reward_sum - parent_reward_sum;
            if(is_reset == 1){
                // parent is reset
                true_reward = node->reward_sum;
            }
            bootstrap_value = true_reward + discount * bootstrap_value;
            min_max_stats.update(bootstrap_value);
        }
    }

    void cmulti_back_propagate(int hidden_state_index_x, float discount, const std::vector<float> &reward_sums,
                               const std::vector<float> &values, const std::vector<std::vector<float>> &policies,
                               tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results,
                               std::vector<int> is_reset_lst, int simulation_idx, const std::vector<std::vector<float>> &gumbels,
                               float c_visit, float c_scale, int simulation_num, int leaf_num){
        for(int i = 0; i < results.num; ++i){

            results.nodes[i]->expand(0, hidden_state_index_x, i, reward_sums[i], policies[i], simulation_num, leaf_num);
            // reset
            results.nodes[i]->is_reset = is_reset_lst[i];
            cback_propagate(results.search_paths[i], min_max_stats_lst->stats_lst[i], 0, values[i], discount);
            CNode* root = results.search_paths[i][0];
            sequential_halving(root, simulation_idx, min_max_stats_lst->stats_lst[i], gumbels[i], c_visit, c_scale, discount);
        }
    }

    float sigma(float value, CNode* root, float c_visit, float c_scale){
        int max_visit = 0;
        for(int a = 0; a < root->action_num; ++a){
            CNode* child = root->get_child(a);
            max_visit = std::max(max_visit, child->visit_count);
        }
        return (c_visit + max_visit) * c_scale * value;
    }

    int cselect_child(CNode* root, tools::CMinMaxStats &min_max_stats, float c_visit, float c_scale, float discount, int simulation_idx, const std::vector<float> &gumbels, int m){
        if (root->is_root == 1) {
            if (simulation_idx == 0) {
                std::vector<std::pair<int, float>> gumbel_policy;
                for(int a = 0; a < root->action_num; ++a){
                    CNode* child = root->get_child(a);
                    gumbel_policy.push_back(std::make_pair(a, gumbels[a] + child->prior));
                }
                sort(gumbel_policy.begin(), gumbel_policy.end(), compare);
                for (int a = 0; a < m; ++a){
                    int to_select = gumbel_policy[a].first;
                    root->selected_children.push_back(std::make_pair(to_select, root->get_child(to_select)));
                }
            }

            std::vector<int> min_index_lst;
            int min_visit = 10000;
            for(int a = 0; a < root->selected_children.size(); ++a){
                CNode* child = root->get_child(root->selected_children[a].first);
                if (child->visit_count < min_visit){
                    min_visit = child->visit_count;
                    min_index_lst.clear();
                    min_index_lst.push_back(root->selected_children[a].first);
                }
            }
            int action = 0;
            if(min_index_lst.size() > 0){
                int rand_index = rand() % min_index_lst.size();
                action = min_index_lst[rand_index];
            }
            return action;
        }
        else {

            float max_score = FLOAT_MIN;
            const float epsilon = 0.000001;
            std::vector<int> max_index_lst;
            std::vector<float> scores = calc_non_root_score(root, min_max_stats, c_visit, c_scale, discount);
            int action = argmax(scores);
            return action;
        }
    }

    int argmax(std::vector<float> arr){
        int index = -3;
        float max_val = FLOAT_MIN;
        for(int i = 0; i < arr.size(); ++i){
            if(arr[i] > max_val){
                max_val = arr[i];
                index = i;
            }
        }
        return index;
    }

    void cmulti_traverse(CRoots *roots, float c_visit, float c_scale, float discount, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, int simulation_idx, const std::vector<std::vector<float>> &gumbels){
        // set seed
        timeval t1;
        gettimeofday(&t1, NULL);
        srand(t1.tv_usec);

        int last_action = -1;
        results.search_lens = std::vector<int>();
        for(int i = 0; i < results.num; ++i){
            CNode *node = &(roots->roots[i]);
            int search_len = 0;
            results.search_paths[i].push_back(node);

            if(DEBUG_MODE){
                printf("=====find=====\n");
            }
            while(node->expanded()){

                int action = cselect_child(node, min_max_stats_lst->stats_lst[i], c_visit, c_scale, discount, simulation_idx, gumbels[i], roots->roots[i].m);
                if(DEBUG_MODE){
                    printf("select action: %d\n", action);
                }
//                printf("total unsigned q: %f\n", total_unsigned_q);
                node->best_action = action;
                // next
                node = node->get_child(action);
                last_action = action;
                results.search_path_actions[i].push_back(action);
                results.search_paths[i].push_back(node);
                search_len += 1;
            }

            CNode* parent = results.search_paths[i][results.search_paths[i].size() - 2];

            results.hidden_state_index_x_lst.push_back(parent->hidden_state_index_x);
            results.hidden_state_index_y_lst.push_back(parent->hidden_state_index_y);

            results.last_actions.push_back(last_action);
            results.first_actions.push_back(results.search_path_actions[i][0]);
            results.search_lens.push_back(search_len);
            results.nodes.push_back(node);
        }
    }


    void cmulti_traverse_return_path(CRoots *roots, float c_visit, float c_scale, float discount, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, int simulation_idx, const std::vector<std::vector<float>> &gumbels){
        // set seed
        timeval t1;
        gettimeofday(&t1, NULL);
        srand(t1.tv_usec);

        int last_action = -1;
//        float parent_q = 0.0;
        results.search_lens = std::vector<int>();
        for(int i = 0; i < results.num; ++i){
            CNode *node = &(roots->roots[i]);
            int search_len = 0;
            results.search_paths[i].push_back(node);

            if(DEBUG_MODE){
                printf("=====find=====\n");
            }
            while(node->expanded()){

//                printf("------------%d---------------\n", node->hidden_state_index_y);
                results.search_path_index_x_lst[i].push_back(node->hidden_state_index_x);
                results.search_path_index_y_lst[i].push_back(node->hidden_state_index_y);
//                printf("-------------------------here----------------------------\n");

                int action = cselect_child(node, min_max_stats_lst->stats_lst[i], c_visit, c_scale, discount, simulation_idx, gumbels[i], roots->roots[i].m);
                if(DEBUG_MODE){
                    printf("select action: %d\n", action);
                }
//                printf("total unsigned q: %f\n", total_unsigned_q);
                node->best_action = action;
                // next
                node = node->get_child(action);
                last_action = action;
                results.search_path_actions[i].push_back(action);
                results.search_paths[i].push_back(node);
                search_len += 1;
            }

            CNode* parent = results.search_paths[i][results.search_paths[i].size() - 2];

            results.hidden_state_index_x_lst.push_back(parent->hidden_state_index_x);
            results.hidden_state_index_y_lst.push_back(parent->hidden_state_index_y);

            results.last_actions.push_back(last_action);
            results.first_actions.push_back(results.search_path_actions[i][0]);
            results.search_lens.push_back(search_len);
            results.nodes.push_back(node);
        }
    }

}