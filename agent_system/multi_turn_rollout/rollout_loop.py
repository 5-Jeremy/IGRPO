# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
import struct
from igpo.core_igpo import compute_answer_block_avg_log_prob

import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data, adjust_batch
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from igrpo.core_igrpo import TrajectoryNodeStateManagement, compute_log_ratio_expand_val
from verl.utils.advantage_estimator import AdvantageEstimator

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
        original_gen_batch_index: np.ndarray = None
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        We may need to connect batch[item] and gen_batch[original_gen_batch_index[item]]
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
            original_gen_batch_index(np.ndarray): when not None, it means that we should construct batch_item
                by gen_batch[original_gen_batch_index[item]]
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """
        gen_batch_index = item if original_gen_batch_index is None else original_gen_batch_index[item]
        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][gen_batch_index]
        data_source = gen_batch.non_tensor_batch['data_source'][gen_batch_index]
        ground_truth = gen_batch.non_tensor_batch['ground_truth'][gen_batch_index]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        
        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        
        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        

        if is_multi_modal:

            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source,
            'ground_truth': ground_truth,
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
        original_gen_batch_index: np.ndarray = None
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
            original_gen_batch_index: the batch we construct may be different order from gen_batch
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
                original_gen_batch_index=original_gen_batch_index
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def gather_rollout_data_tree_structure(
            self,
            total_batch_list: List[Dict],
            success: Dict[str, np.ndarray],
            global_steps: int,
            ) -> DataProto:
        """
        Returns:
            DataProto: Collected and organized tree-structured trajectory data
        """

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)

        for data in total_batch_list:
            # add success_rate
            for key, value in success_rate.items():
                data[key] = value
            # add info gain to reward for those deactivate nodes
            if data['deactivate'] == True:
                assert data['traj_step'] < self.config.env.max_steps - 1
                # balance between exploration and exploitation
                multi = 0.5 if global_steps <= self.config.algorithm.igrpo.stable_steps else 1.0
                base = data['info_gain_sum']
                if global_steps > self.config.algorithm.igrpo.stable_steps and self.config.algorithm.igrpo.stable_method == 'threshold':
                    base = np.float32(1.0) if base >= 0.5 else np.float32(0.0)
                data['rewards'] = base * multi
                
        if self.config.algorithm.igrpo.reward_mode == 'full':
            # From every terminal_node, we go up to obtain a full trajectory.
            # The reward manager is chain-based manager, since we obtain many paths here.
            batch_list: list[dict] = []
            extra_info_list: list[dict] = []
            node_uid2index = {}
            for i, data in enumerate(total_batch_list):
                node_uid2index[data['node_uid']] = i
            for i, data in enumerate(total_batch_list):
                if data['is_terminal'] == True:
                    traj_uid = str(uuid.uuid4())
                    reward = data['rewards']
                    episode_length = np.float32(data['traj_step'] + 1)
                    current_tool_calling = data['current_tool_callings']
                    node_uid = data['node_uid']
                    index = i
                    while index is not None:
                        batch_list.append(total_batch_list[index])
                        extra_info = {}
                        extra_info['traj_uid'] = traj_uid
                        extra_info['episode_rewards'] = reward
                        extra_info['episode_lengths'] = episode_length
                        extra_info['tool_callings'] = current_tool_calling
                        extra_info_list.append(extra_info)
                        node_uid = total_batch_list[index]['parent_node_uid']
                        index = node_uid2index.get(node_uid, None)
        else:
            # The reward manager is tree-structure manager.
            batch_list: list[dict] = total_batch_list
            extra_info_list: list[dict] = None
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(batch_list, extra_info_list)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        batch_size = len(gen_batch.batch)

        # Initial observations from the environment
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info

            # pad to be divisible by dp_size
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            # # unpad
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid
            batch.non_tensor_batch['traj_step'] = [_step] * batch_size

            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            batch.non_tensor_batch['text_actions'] = text_actions

            next_obs, rewards, dones, infos = envs.step(text_actions)

            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Create reward tensor, only assign rewards for active environments
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            # Update episode lengths for active environments
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
    
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list, 
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings

    def vanilla_multi_turn_loop_with_tree_structure(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop. some trajectories share the same prefix,
        thus forming a tree structure.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (Dict): List of trajectory node data, each representing a node in the trajectory tree
        """

        batch_size = len(gen_batch.batch)

        # Initial observations from the environment
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        
        node_management = TrajectoryNodeStateManagement(batch_size, envs=envs, obs=obs)
        node_management.assign_group_uids(self.config.env.rollout.n)

        webshop_scorer = None
        webshop_deferred = None
        if self.config.algorithm.adv_estimator == AdvantageEstimator.TREEHCA:
            from treehca.training_webshop_scoring import WebshopInfoGainScorer, resolve_treehca_scorer

            if resolve_treehca_scorer(self.config) == "webshop":
                if not hasattr(envs, "scoring_payloads"):
                    raise ValueError("TreeHCA webshop scoring requires TreeHCAWebshopEnvironmentManager")
                webshop_scorer = WebshopInfoGainScorer(
                    self.tokenizer, actor_rollout_wg,
                    max_model_len=self.config.actor_rollout_ref.rollout.get("max_model_len") or (self.config.data.max_prompt_length + self.config.data.max_response_length),
                    path_probability_threshold=self.config.algorithm.treehca.get("webshop_path_probability_threshold", 1e-3),
                    batch_size=self.config.algorithm.treehca.get("webshop_probe_batch_size", 32),
                    prune_unsuccessful_choices=self.config.algorithm.treehca.get("webshop_prune_unsuccessful_choices", True),
                    apply_chat_template_kwargs=self.config.data.get("apply_chat_template_kwargs", {}),
                )
                from treehca.webshop_deferred_scores import WebshopDeferredScores

                webshop_deferred = WebshopDeferredScores(
                    node_management.uid_batch,
                    prob_diff_mode=self.config.algorithm.igrpo.prob_diff_mode,
                    prob_floor=self.config.algorithm.treehca.prob_floor,
                )
        # A scorer lives for one rollout collection, while the policy is fixed.
        total_batch_list = []
        node_uid2info_gain_sum = {}
        total_infos = []
        # Tree Structured Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = deepcopy(node_management.active_nodes) # neccessary to deepcopy
            is_last_step = _step == self.config.env.max_steps - 1
    
            batch = self.preprocess_batch(gen_batch=gen_batch, 
                                          obs=node_management.obs, 
                                          original_gen_batch_index=node_management.original_gen_batch_index)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info

            # pad to be divisible by dp_size
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            # # unpad
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['parent_node_uid'] = node_management.node_uid
            batch.non_tensor_batch['node_uid'] = node_management.update_node_uid()
            batch.non_tensor_batch['uid'] = node_management.uid_batch
            batch.non_tensor_batch['traj_step'] = np.array([_step] * batch_size, dtype=np.int32)

            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            batch.non_tensor_batch['text_actions'] = text_actions

            previous_page_types = node_management.envs.scoring_page_types() if webshop_scorer is not None else None
            if self.config.algorithm.adv_estimator == AdvantageEstimator.TREEHCA:
                # This is the page in the prompt, before the action changes the environment.
                batch.non_tensor_batch["page_type"] = np.asarray(previous_page_types if previous_page_types is not None else [None] * batch_size, dtype=object)
            next_obs, rewards, dones, infos = node_management.envs.step(text_actions)
            if self.config.algorithm.adv_estimator == AdvantageEstimator.TREEHCA:
                # is_terminal can alias dones and is later updated for pruning.
                logging_dones = np.asarray(dones, dtype=bool).reshape(-1).copy()
            node_management.obs = next_obs

            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            if 'tool_calling' in infos[0]:
                node_management.tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            
            # Create reward tensor, only assign rewards for active environments
            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=False)
            batch.non_tensor_batch['current_tool_callings'] = node_management.tool_callings
            batch.non_tensor_batch['is_terminal'] = torch_to_numpy(dones, is_object=False)
            if is_last_step:
                batch.non_tensor_batch['is_terminal'] = np.ones_like(batch.non_tensor_batch['is_terminal'], dtype=np.bool_)

            # compute info gain
            with torch.no_grad():
                info_gain_batch = self.preprocess_batch(gen_batch=gen_batch, 
                                            obs=node_management.obs, 
                                            original_gen_batch_index=node_management.original_gen_batch_index)
                info_gain_batch.batch["prompts"] = info_gain_batch.batch.pop("input_ids")
                if webshop_scorer is not None:
                    info_gain_batch.non_tensor_batch.update(
                        webshop_scoring_payload=node_management.envs.scoring_payloads(next_obs["text"], previous_page_types, active_masks, dones),
                        webshop_active=np.asarray(active_masks, dtype=bool),
                        webshop_terminated=np.asarray(dones, dtype=bool),
                        webshop_won=np.asarray([info["won"] for info in infos], dtype=bool),
                    )
                    # The adapter pads its generated probes, not observation slots.
                    reorder_index = torch.arange(batch_size)
                    info_gain_batch = webshop_scorer.compute(info_gain_batch, policy_version=0)
                    webshop_scorer.require_active_scores(info_gain_batch)
                    batch.non_tensor_batch["webshop_success_probability"] = info_gain_batch.non_tensor_batch["webshop_success_probability"]
                    batch.non_tensor_batch["webshop_skipped_reason"] = info_gain_batch.non_tensor_batch["webshop_skipped_reason"]
                    # Only inactive rows can still be NaN after validation.
                    info_gain_batch.batch["avg_ans_log_probs"].nan_to_num_(nan=0.0, neginf=-float("inf"))
                    # Preserve raw log scores: deferred means are computed in log space.
                    batch.non_tensor_batch["avg_ans_log_probs"] = torch_to_numpy(info_gain_batch.batch["avg_ans_log_probs"], is_object=False).copy()
                    if not self.config.algorithm.igrpo.prob_diff_mode:
                        info_gain_batch.batch["avg_ans_log_probs"].clamp_(min=np.log(self.config.algorithm.treehca.prob_floor))
                else:
                    info_gain_batch, reorder_index = adjust_batch(config=self.config, data=info_gain_batch, mode="copy", info_gain_compute=True)
                    info_gain_batch = compute_answer_block_avg_log_prob(batch=info_gain_batch,
                                                                        tokenizer=self.tokenizer,
                                                                        actor_rollout_wg=actor_rollout_wg,
                                                                        think=False,
                                                                        response_length=self.config.data.max_response_length)
                avg_ans_log_probs = info_gain_batch.batch.pop("avg_ans_log_probs")
                del info_gain_batch
                if self.config.algorithm.adv_estimator == AdvantageEstimator.TREEHCA and webshop_scorer is None:
                    batch.non_tensor_batch["avg_ans_log_probs"] = torch_to_numpy(avg_ans_log_probs[torch.argsort(reorder_index)], is_object=False)[:batch_size].copy()
                info_gain_sum = torch.exp(avg_ans_log_probs) if self.config.algorithm.igrpo.prob_diff_mode else avg_ans_log_probs
                del avg_ans_log_probs
                info_gain_sum = info_gain_sum[torch.argsort(reorder_index)]
                del reorder_index
                info_gain_sum = torch_to_numpy(info_gain_sum, is_object=False)
            torch.cuda.empty_cache()
            info_gain_sum = info_gain_sum[:batch_size]
            batch.non_tensor_batch["info_gain_sum"] = info_gain_sum
            batch.non_tensor_batch["info_gain"] = np.zeros_like(info_gain_sum)
            parent_info_gain_sum_batch = np.zeros_like(info_gain_sum)
            is_root_parent_batch = np.ones(batch_size, dtype=np.bool_)
            if webshop_deferred is not None:
                webshop_deferred.update(batch, active_masks, node_uid2info_gain_sum)
                for i in range(batch_size):
                    if not active_masks[i]:
                        continue
                    parent_node_uid = batch.non_tensor_batch["parent_node_uid"][i]
                    if parent_node_uid != "root":
                        assert parent_node_uid in node_uid2info_gain_sum, (
                            "Missing key in node_uid2info_gain_sum: "
                            f"{parent_node_uid}"
                        )
                        parent_info_gain_sum_batch[i] = (
                            node_uid2info_gain_sum[parent_node_uid]
                        )
                        is_root_parent_batch[i] = False
                    else:
                        # compute_log_ratio_expand_val treats root parents as prob_floor,
                        # so the value in parent_info_gain_sum_batch is ignored.
                        parent_info_gain_sum_batch[i] = 0.0
            else:
                for i in range(batch_size):
                    if active_masks[i]:
                        info_gain_sum = batch.non_tensor_batch["info_gain_sum"][i]
                        parent_node_uid = batch.non_tensor_batch['parent_node_uid'][i]
                        if parent_node_uid != "root":
                            assert parent_node_uid in node_uid2info_gain_sum, f"Missing key in node_uid2info_gain_sum: {parent_node_uid}"
                            parent_info_gain_sum = node_uid2info_gain_sum[parent_node_uid]
                            is_root_parent_batch[i] = False
                        else:
                            # we assume that without any search, info gain is 0.
                            parent_info_gain_sum = 0.0
                        batch.non_tensor_batch["info_gain"][i] = info_gain_sum - parent_info_gain_sum
                        parent_info_gain_sum_batch[i] = parent_info_gain_sum

            # dynamically change the node_management, according to the info gain
            node_management.deactivate(torch_to_numpy(dones, is_object=False))
            info_gain_sum = batch.non_tensor_batch["info_gain_sum"]
            info_gain = batch.non_tensor_batch["info_gain"]
            expand_score = self.config.algorithm.igrpo.expand_score
            if expand_score == 'info_val':
                info_val = (info_gain_sum + info_gain) / 2.0
            elif expand_score == 'log_ratio':
                info_val = compute_log_ratio_expand_val(
                    info_gain_sum=info_gain_sum,
                    parent_info_gain_sum=parent_info_gain_sum_batch,
                    is_root_parent=is_root_parent_batch,
                    prob_diff_mode=self.config.algorithm.igrpo.prob_diff_mode,
                    prob_floor=self.config.algorithm.igrpo.expand_prob_floor,
                    dtype=info_gain_sum.dtype,
                )
            else:
                raise ValueError(f"Invalid expand_score: {expand_score}, expected one of ['info_val', 'log_ratio']")
            
            branchable_mask = None
            if "webshop" in self.config.env.env_name.lower():
                # ``infos`` describes next_obs, which is the observation from
                # which the next action (and any forked siblings) would start.
                # Item subpages have only a single useful continuation, so keep
                # that path alive without ever duplicating it.
                branchable_mask = np.asarray(
                    [info.get("page_type") != "item_sub_page" for info in infos],
                    dtype=np.bool_,
                )

            expand_prob = node_management.compute_expand_prob(val=info_val, gamma=self.config.algorithm.igrpo.gamma)
            if branchable_mask is not None:
                # Expansion probabilities describe selection for *additional*
                # continuations. Remove non-branchable nodes and renormalize
                # each prompt group over its eligible nodes.
                expand_prob[node_management.active_nodes & ~branchable_mask] = 0
                branchable_groups = np.unique(
                    node_management.uid_batch[node_management.active_nodes & branchable_mask]
                )
                for group_uid in branchable_groups:
                    group_mask = (
                        node_management.active_nodes
                        & branchable_mask
                        & (node_management.uid_batch == group_uid)
                    )
                    probability_sum = expand_prob[group_mask].sum()
                    if probability_sum > 0:
                        expand_prob[group_mask] /= probability_sum
                    else:
                        expand_prob[group_mask] = 1 / np.count_nonzero(group_mask)
            expand_num = node_management.get_expand_num(expand_prob=expand_prob,
                                                        max_traj_to_expand_per_node=self.config.algorithm.igrpo.max_traj_to_expand_per_node,
                                                        expand_mode=self.config.algorithm.igrpo.expand_mode,
                                                        branchable_mask=branchable_mask)
            # those activate nodes now deactivate here, since they don't expand.
            batch.non_tensor_batch['deactivate'] = (~batch.non_tensor_batch['is_terminal']) & (expand_num == 0)
            # those current activate nodes, since they don't expand, so they are terminal and deactivated.
            batch.non_tensor_batch['is_terminal'] |= batch.non_tensor_batch['deactivate']

            if self.config.algorithm.adv_estimator == AdvantageEstimator.TREEHCA:
                from treehca.rollout_records import capture_branch_logging

                batch.non_tensor_batch.update(capture_branch_logging(
                    next_obs=next_obs, infos=infos, dones=logging_dones, is_last_step=is_last_step,
                    expand_prob=expand_prob, expand_num=expand_num, branch_score=info_val,
                    gamma=self.config.algorithm.igrpo.gamma,
                ))
            
            node_management.deactivate(expand_num == 0)

            batch_list: list[dict] = to_list_of_dict(batch)
            if webshop_deferred is not None:
                webshop_deferred.register_rows([batch_list[i] for i in range(batch_size) if active_masks[i]])
            for i in range(batch_size):
                if active_masks[i]:
                    total_batch_list.append(batch_list[i])
                    total_infos.append(infos[i])
                    node_uid2info_gain_sum[batch_list[i]['node_uid']] = batch_list[i]['info_gain_sum']

            # [WARNING!] we only change the node_management(fork from) AFTER to_list_of_dict(batch)
            if not np.any(node_management.active_nodes) or is_last_step:
                break
            node_management.expand(expand_num)
            
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    tree_structure=True
                    )
        self._webshop_scorer_metrics = webshop_scorer.metrics() if webshop_scorer is not None else None
        return total_batch_list, success

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
        # TreeHCA branches exactly like IGRPO and only differs in credit assignment
        tree_structure_traj = self.config.algorithm.adv_estimator in (AdvantageEstimator.IGRPO, AdvantageEstimator.TREEHCA) and is_train
            
        # Initial observations from the environment
        if tree_structure_traj:
            total_batch_list, total_success = \
                self.vanilla_multi_turn_loop_with_tree_structure(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        elif self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:
            # Vanilla Sampling   
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        
        # for tree structure trajectory, we only return total_batch_list and total_success, where each item in batch_list is a node in the tree.
        if not tree_structure_traj:
            assert len(total_batch_list) == len(total_episode_rewards)
            assert len(total_batch_list) == len(total_episode_lengths)
            assert len(total_batch_list) == len(total_traj_uid)
            assert len(total_batch_list) == len(total_tool_callings)
            # Create trajectory data
            gen_batch_output: DataProto = self.gather_rollout_data(
                total_batch_list=total_batch_list,
                episode_rewards=total_episode_rewards,
                episode_lengths=total_episode_lengths,
                success=total_success,
                traj_uid=total_traj_uid,
                tool_callings=total_tool_callings,
            )
        else:
            gen_batch_output: DataProto = self.gather_rollout_data_tree_structure(
                total_batch_list=total_batch_list,
                success=total_success,
                global_steps=gen_batch.meta_info["global_steps"]
            )
            if self._webshop_scorer_metrics is not None:
                gen_batch_output.meta_info["webshop_scorer_metrics"] = self._webshop_scorer_metrics
        
        return gen_batch_output
