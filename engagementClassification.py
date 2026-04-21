import numpy as np
from collections import defaultdict, Counter
import openai
from engageConfig import *



class EngagementClassifier:
    def __init__(self,action_names,usedLLM='gpt',contextAware=True):

        self.context_aware = contextAware
        # self.used_class_names = ['Writing on notebook/tablet', 'Typing on a laptop', 'Playing with mobile phone', 'Reading',
        #                     'Raising hand', 'Drinking', 'Eating meal/snack', "Checking watch", 'Yawning',
        #                     'Looking to the side/back', 'Listening', 'Looking down without reading/writing',
        #                     'Looking at laptop screen (not typing)']

        self.used_class_names = action_names
        self.client = None
        self.tokenizer = None
        self.model = None
        self.terminators = None
        self.usedLLM = usedLLM
        if usedLLM == 'llama':
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto")
            self.terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        elif usedLLM == 'gemma':
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
            self.model = AutoModelForCausalLM.from_pretrained(
                "google/gemma-2-9b-it",
                device_map="auto",
                torch_dtype=torch.bfloat16)

        self.annotate_fn = self.annotate
        if usedLLM == 'llama':
            self.annotate_fn = self.annotate_llama
        elif usedLLM == 'gemma':
            self.annotate_fn = self.annotate_gemma

        self.system_context_aware_message =f"""
        You are an expert educational behavior analyst specializing in classroom observation and learning analytics. 
        You are given: Student_Sequence: a sequence of the student’s time-stamped actions over a fixed time interval.
        Peers_Aggregate: the aggregate (majority) sequence of peers’ actions over the same interval, representing classroom context.
        Your task is to classify the student’s behavioral engagement level into one of the following categories:
        High engagement: Predominantly learning-oriented behaviors, either active or passive (e.g., Writing on notebook/tablet, Typing for class-related work, Reading, Listening attentively, Asking or answering questions, Pointing to instructional material).
        Low engagement: Predominantly distracted or disengaged behaviors, with only brief or sporadic moments of on-task activity (e.g., Playing with a mobile phone, off-task laptop use, repeated checking of personal items, extended inattentive behavior).
        Decision Guidelines
        1-Interpretation of ambiguous actions
    	    -Treat “Looking at laptop screen (not typing)” as a reading action only if it is interspersed between clearly engaged actions (e.g., Writing, Typing, Listening attentively). Otherwise, consider it potentially off-task.
        2- Primary evidence: student behavior
            -Prioritize the student’s own actions and their durations when determining engagement.
            -Sustained on-task behavior outweighs brief off-task interruptions.
        3-Secondary evidence: peer context
            -Use Peers_Aggregate as supporting evidence, not the primary determinant.
            -If the student’s actions slightly align with the majority of the class (not necessarily at every time segment), this reinforces engagement.
            -If the student’s actions consistently diverge from peers’ instructional activities, this may slightly lower engagement, but only insofar as it helps interpret ambiguous student actions.
        4-Overall judgment
            -Base the final label on the dominant pattern across the interval, not isolated moments.
            -Consider whether the student’s behavior plausibly supports learning given the classroom context.
        Provide your response as a Python dictionary string with 'Category' and 'reasoning' keys. Do not repeat or return the input.
        """
        self.system_message =f""" You are an educational behavior analyst. You are given a sequence of time-stamped student's actions in a classroom during 120-second interval. 
                                Your task is to classify the student’s overall behavioral engagement level into one of the following categories:
                                 High engagement: Focused and active learning behaviors (e.g., taking notes, listening attentively, asking/answering questions, typing,pointing to something).
                                 Low engagement: Frequently distracted or disengaged with only brief moments of on-task behavior.
                                 Instructions:
                                 1-if multiple actions are present, weigh them by duration. 
                                 2-The action 'looking at laptop screen (not typing)' can be treated as a reading action if it is interspersed between engaged actions (e.g., writing, typing, listening attentively).
                                 3- The longer-lasting behaviors should influence the classification more.
                                Provide both the classification and a brief reasoning.
                                The generated response should be in the form of a Python dictionary string with key 'Category' for engagement category and 'reasoning' for the reasoning of your choice.
                                Please do not repeat or return the content back again.
                                    """
    def classify_action_sequence(self,actions_pred_dict):
        sts_actions_at_time_interval = self.get_students_actions_preds_at_time_interval(actions_pred_dict)
        sts_actions = {st_id: _actions[0] for st_id, _actions in sts_actions_at_time_interval.items()}
        sts_actions_str_representation = {st_id: st_actions[1] for st_id, st_actions in
                                          sts_actions_at_time_interval.items()}

        sts_engagement_levels_dict ={}
        if self.context_aware:
            classroom_context = self.aggregate_student_actions(sts_actions, bin_size=10, method="majority")
            peers_action_seqs = sts_actions_str_representation
            for track_id, st_actions in sts_actions_str_representation.items():
                response = self.annotate_fn(st_actions, classroom_context)
                print(track_id, response)
                if self.usedLLM == 'gemma':
                    pred_class = response.replace("\n", "").split('model')[1].split(':')[1].split(',')[0][2:-1]
                else:
                    pred_class = response.replace("\n", "").split(',')[0].split(':')[-1][2:-1]
                pred_class = pred_class.lower()
                sts_engagement_levels_dict[track_id] = pred_class
        else:
            for track_id, st_actions in sts_actions_str_representation.items():
                response = self.annotate_fn(st_actions)
                print(track_id, response)
                if self.usedLLM == 'gemma':
                    pred_class = response.replace("\n", "").split('model')[1].split(':')[1].split(',')[0][2:-1]
                else:
                    pred_class = response.replace("\n", "").split(',')[0].split(':')[-1][2:-1]
                pred_class = pred_class.lower()
                sts_engagement_levels_dict[track_id] = pred_class

        return sts_engagement_levels_dict

    def annotate(self,student_actions, classroom_context=''):

        if self.context_aware:
            sys_mes = self.system_context_aware_message
            user_message = f"Student actions with start and end times (in 'MM:SS' format): {student_actions}. the aggregate sequence of actions from their peers with start and end times (in 'MM:SS' format): {classroom_context}"
        else:
            sys_mes = self.system_message
            user_message = f"The student's action(s) with start and end times (in 'MM:SS' format) is/are listed as follows: {student_actions}."

        # Generate completion with OpenAI GPT-3
        response_data = None
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",

                        "content": sys_mes
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }

                ],
                temperature=0.1,
            )
            try:
                response_data = completion.choices[0].message.content

            except Exception as e:
                print('erro')

        except Exception as e:
            print(f"An error occurred: {e}. Skipping this item.")
            response_dict = {"error": "An unknown error occurred."}

        return response_data

    def annotate_llama(self,student_actions, classroom_context=''):
        if self.context_aware:
            sys_mes = self.system_context_aware_message
            user_message = f" Student actions with start and end times (in 'MM:SS' format): {student_actions}. the aggregate sequence of actions from his peers with start and end times (in 'MM:SS' format): {classroom_context}"
        else:
            sys_mes = self.system_message
            user_message =f"The student's action(s) with start and end times (in 'MM:SS' format) is/are listed as follows: {student_actions}."

        messages = [
            {"role": "system", "content": sys_mes},
            {"role": "user", "content": user_message}
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        response_data = None
        try:
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id=self.terminators,
                # attention_mask=attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.1,
            )

            response = outputs[0][input_ids.shape[-1]:]
            response_data = self.tokenizer.decode(response, skip_special_tokens=True)

        except Exception as e:
            print(f"An error occurred: {e}. Skipping this item.")
            response_data = {"error": "An unknown error occurred."}
        return response_data
    def aggregate_student_actions(self,sts_actions_at_time_interval, bin_size=10, method="majority"):
        """
        Aggregate time-stamped actions of students into classroom-level summaries.

        Args:
            sts_actions_at_time_interval (dict): {student_id: [(start, end, action), ...]}
            bin_size (int): size of time bins in seconds
            method (str): "majority" or "distribution"

        Returns:
            pd.DataFrame: time bins with aggregate action summaries
        """
        students_actions = []
        for seq_name, st_actions in sts_actions_at_time_interval.items():
            students_actions.extend(st_actions)
        # Step 1: find global timeline
        max_time = 0
        for act in students_actions:
            end = act['end_time']
            max_time = max(max_time, end)

        # Step 2: normalize into bins
        bins = list(range(0, max_time + bin_size, bin_size))
        results = ''

        for i in range(len(bins) - 1):
            start, end = bins[i], bins[i + 1]
            actions_in_bin = []

            # collect each student's action during this bin
            for action in students_actions:
                s = action['start_time']
                e = action['end_time']
                action_label = action['action_label']
                # check if bin overlaps with student's action interval
                # condition of not overlapping is as follows: start_interval_1>= end_interval_2 or end_interval_1 <= start_interval_2===#
                if not (e <= start or s >= end):
                    actions_in_bin.append(action_label)

            if not actions_in_bin:
                continue

            if method == "majority":
                most_common_action, freq = Counter(actions_in_bin).most_common(1)[0]

                m_start, s_start = divmod(start, 60)
                m_end, s_end = divmod(end, 60)
                # results = results +" most students "+ most_common + f'({start}-{end}s)' + ';'
                results = results + " most students " + most_common_action + f'({m_start:02d}:{s_start:02d}-{m_end:02d}:{s_end:02d})' + ';'
                ###results = results + f"{round(freq/len(actions_in_bin)*100)}% of students " + most_common_action + f'({m_start:02d}:{s_start:02d}-{m_end:02d}:{s_end:02d})' + ';'
                # results.append({"start": start, "end": end, "summary": most_common})

            elif method == "distribution":
                counts = Counter(actions_in_bin)
                total = sum(counts.values())
                dist_summary = "; ".join([f"{round(v / total * 100)}% {k}" for k, v in counts.items()])
                # results.append({"start": start, "end": end, "summary": dist_summary})

        return results[:-1]

    def get_students_actions_preds_at_time_interval(self,actions_pred_dict):
        sts_actions_seq_dict ={}
        starts, ends = self.get_starts_ends_actions()
        for track_id, action_lst in actions_pred_dict.items():
            st_actions_preds_str = ','.join(action_lst)
            print(st_actions_preds_str)
            st_actions_lst, st_action_seq = self.get_student_actions_timestamps(st_actions_preds_str, starts, ends,
                                                                                 returnActionLst=True)
            sts_actions_seq_dict[track_id] = [st_actions_lst, st_action_seq]

        return sts_actions_seq_dict

    def get_student_actions_timestamps(self,st_action_preds, pred_time_starts, pred_time_ends, rep_format='sequence',
                                       returnActionLst=False):

        st_action_seq = ''
        st_actions_lst = []
        st_action_preds = st_action_preds.strip().split(',')
        current_action = st_action_preds[0]
        start_time = pred_time_starts[0]
        m_start, s_start = divmod(round(start_time), 60)

        st_action_histogram = {label: 0 for label in self.used_class_names}
        for i in range(1, len(st_action_preds)):

            if current_action != st_action_preds[i]:
                end_time = pred_time_starts[i]
                m_end, s_end = divmod(round(end_time), 60)
                st_action_seq = st_action_seq + current_action + f'({m_start:02d}:{s_start:02d}-{m_end:02d}:{s_end:02d})' + ';'
                st_actions_lst.append(
                    {'action_label': current_action, 'start_time': round(start_time), 'end_time': round(end_time)})
                st_action_histogram[current_action] += (round(end_time) - round(start_time))

                if i == len(st_action_preds) - 1:
                    end_time = pred_time_ends[i]
                    m_end, s_end = divmod(round(end_time), 60)
                    st_action_seq = st_action_seq + st_action_preds[
                        i] + f'({m_start:02d}:{s_start:02d}-{m_end:02d}:{s_end:02d})' + ';'
                    st_actions_lst.append(
                        {'action_label': st_action_preds[i], 'start_time': round(start_time),
                         'end_time': round(end_time)})

                    st_action_histogram[st_action_preds[i]] += (round(end_time) - round(start_time))
                else:
                    current_action = st_action_preds[i]
                start_time = end_time
                m_start, s_start = divmod(round(start_time), 60)

            else:
                # === change end time of the current action to match the end time of vid-annot
                if i == len(st_action_preds) - 1:
                    end_time = pred_time_ends[i]
                    m_end, s_end = divmod(round(end_time), 60)
                    st_action_seq = st_action_seq + current_action + f'({m_start:02d}:{s_start:02d}-{m_end:02d}:{s_end:02d})' + ';'
                    st_actions_lst.append(
                        {'action_label': current_action, 'start_time': round(start_time),
                         'end_time': round(end_time)})
                    st_action_histogram[current_action] += (round(end_time) - round(start_time))

        if rep_format == 'histogram':
            st_actions = ''
            for label, duration in st_action_histogram.items():
                st_actions = st_actions + label + ' : ' + str(duration) + ';'
            return st_actions[:-1]
        if returnActionLst:
            return st_actions_lst, st_action_seq
        return st_action_seq

    def get_starts_ends_actions(self):
        # === INPUT PARAMETERS ===
        fps = 15  # frames per second
        window_size = 45  # frames per prediction window
        stride = 45  # frames between consecutive windows (e.g., 50% overlap)
        num_frames = 1800  # total frames in the video (example: 2 minutes at 15 fps)
        # === DERIVED PARAMETERS ===
        frame_duration = 1 / fps
        window_duration = window_size / fps
        stride_duration = stride / fps
        # === Generate timeline ===
        # The number of steps is floor((num_frames - window_size) / stride) + 1
        num_steps = int(np.floor((num_frames - window_size) / stride)) + 1
        starts = np.arange(num_steps) * stride_duration
        ends = starts + window_duration
        return (starts, ends)