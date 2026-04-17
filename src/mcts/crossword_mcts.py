import re
import os
import json
import numpy as np
import copy

from src.tot.tasks.base import Task, DATA_PATH
from src.tot.prompts.crosswords import cot_prompt, chat_cot_prompt_small


class CrosswordsEnv:
    """
    Environment for 5x5 mini crossword puzzles.

    Each task consists of 5 horizontal clues and 5 vertical clues.
    The model generates letter-by-letter answers which are evaluated
    against the ground-truth board.
    """

    def __init__(self, file="../tot/data/crosswords/mini0505.json", reward_type='reward_rule'):
        self.file = file
        self.file = json.load(open(self.file))
        self.task_num = len(self.file)
        self.reward_type = reward_type
        self.env_name = 'crosswords'

        self.task_inputs = [data[0] for data in self.file]
        self.task_answers = [data[1] for data in self.file]

    def reset_random(self):
        self.task_id = np.random.randint(0, self.task_num)
        self.current_input = self.task_inputs[self.task_id]
        self.current_answer = self.task_answers[self.task_id]

    def reset(self, task_id):
        self.task_id = task_id
        self.current_input = self.task_inputs[self.task_id]
        self.current_answer = self.task_answers[self.task_id]

    def get_input_data(self, state, num=5, stop_endline=True):
        input_data = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a clever AI Assistant which carefully follow the instruction to solve the problem.",
                },
                {"role": "user", "content": self.get_whole_prompt(state=state)},
            ],
            "max_tokens": 500,
            "temperature": 0.7,
            "n": num,
            "stop": '\n' if stop_endline else [],
        }
        return input_data

    def get_whole_prompt(self, state=''):
        input = self.get_input()
        return chat_cot_prompt_small.format(input=input, state=state)

    def get_input(self):
        section = ""
        for i, item in enumerate(self.current_input):
            if i < 5:
                section += f"h{i + 1}. {item}\n"
            else:
                section += f"v{i - 4}. {item}\n"
        section += "Thoughts:"
        return section

    def __len__(self):
        return self.task_num

    def reward(self, output_raw: str):
        output = output_raw.split('Output:\n')[-1]
        letters = []
        for i, line in enumerate(output.strip().split('\n')[-5:], 1):
            letters_line = line.split(' ')[:5]
            letters_line += [' '] * (5 - len(letters_line))
            letters.extend(letters_line)

        letters = letters + [' '] * (25 - len(letters))

        if len(letters) != 25 or len(self.current_answer) != 25:
            print('Warning: unexpected output length in reward computation')
            return 0.0

        reward_letter = 0
        reward_word = 0

        for i in range(0, len(letters), 5):
            if letters[i:i+5] == self.current_answer[i:i+5]:
                reward_word += 1
        for i in range(5):
            if letters[i:25:5] == self.current_answer[i:25:5]:
                reward_word += 1
        reward_word = reward_word / 10

        for i in range(25):
            if letters[i] == self.current_answer[i]:
                reward_letter += 1
        reward_letter = reward_letter / 25

        reward_map = {
            'reward_letter': reward_letter,
            'reward_word': reward_word,
            'reward_rule': reward_letter,
        }
        return reward_map[self.reward_type]

    def answered(self, output: str):
        if "Output:\n" in output:
            if len(output.strip().split('Output:\n')[-1].split('\n')) == 5:
                return True
        return False

    def get_ans(self, board):
        ans = [''] * 10
        for i in range(5):
            ans[i] = ''.join(board[i*5:(i+1)*5])
        for i in range(5):
            ans[i+5] = ''.join(board[i::5])
        return ans

    def render_clues(self, status=None):
        s = ""
        for i in range(5):
            if status is None or self.status[i] == status:
                s += 'h' + str(i+1) + '. ' + self.data[i] + '\n'
        for i in range(5, 10):
            if status is None or self.status[i] == status:
                s += 'v' + str(i-5+1) + '. ' + self.data[i] + '\n'
        return s


class MiniCrosswordsTask(Task):
    """
    Tree-of-Thoughts task wrapper for the 5x5 mini crossword environment.

    Input  (x): Description of a 5x5 mini crossword (10 clues)
    Output (y): 10 five-letter words filling the grid
    Reward (r): Word-level and letter-level accuracy
    """

    def __init__(self, file):
        super().__init__()
        self.env = CrosswordsEnv(file)
        self.xs = []
        for idx in range(len(self.env)):
            self.env.reset(idx)
            self.xs.append(self.env.render_clues())
        self.steps = 10
        self.cache_proposals = {}

    def __len__(self) -> int:
        return len(self.env)

    def set_status(self, x: str, y: str):
        idx = self.xs.index(x)
        self.test_output(idx, y)

    def get_input(self, idx: int) -> str:
        self.env.reset(idx)
        return self.env.render_clues()
