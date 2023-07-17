import os
import re
import copy
import numpy as np
from typing import Callable

from modules.scripts import basedir


class PresetInstruction:
    re_i_calculation = re.compile(r"""
    ^
        (?P<operator>[+\-*/]) # operator
        \s?
        (?P<value>-?(?:\d+\.\d*|\d*\.\d+|\d+)) # integer or float
    $
    """, re.VERBOSE)
    re_i_set = re.compile(r'^set\s(?P<params>.+)$')
    re_i_flag_operation = re.compile(r"""
    ^
        (?P<keyword>only|include|exclude|invert) # keyword: only, include, exclude, invert
        \s
        (?P<params>.+) # parameters
    $
    """, re.IGNORECASE | re.VERBOSE)
    re_i_output = re.compile(r'^output$', re.IGNORECASE)
    re_i_save = re.compile(r'^save\((?P<params>.*)\)$', re.IGNORECASE)

    re_c_expression = re.compile(r'^(?P<targets>(.+?))\s?(?P<operator>[+\-*/]?=)\s?(?P<value>(.+))$')

    re_s_all = re.compile(r'^all(?:\(except\s(?P<params>.+)\))?$', re.IGNORECASE)
    re_s_multiple_blocks = re.compile(r'^(?P<block_a>[^(]+)-(?P<block_b>[^)]+)$')
    re_s_included_or_excluded = re.compile(r'^(?P<targets>included|excluded)$', re.IGNORECASE)
    re_s_condition = re.compile(r"""
    ^
        (?P<operator>>|>=|<|<=|==|!=) # operator: >, >=, <, <=, ==, !=
        \s?
        (?P<value>-?(?:\d+\.\d*|\d*\.\d+|\d+)) # integer or float
    $
    """, re.VERBOSE)
    re_s_group = re.compile(r'^\((?P<targets>.+)\)$')

    re_v_integer_or_float = re.compile(r'^(?P<value>(-?)(\d+\.\d*|\d*\.\d+|\d+))$')
    re_v_rand_r_or_u = re.compile(r'^(?P<rand_type>[ru])$', re.IGNORECASE)
    re_v_rand_x = re.compile(r"""
    ^
        x\(
            (?P<rmin>-?(?:\d+\.\d*|\d*\.\d+|\d+))? # rmin: integer or float
            (?:;(?P<rmax>-?(?:\d+\.\d*|\d*\.\d+|\d+)))? # rmax: integer or float
            (?:;(?P<step>-?(?:\d+\.\d*|\d*\.\d+|\d+)))? # step: integer or float
        \)
    $
    """, re.IGNORECASE | re.VERBOSE)

    re_r_whitespace = re.compile(r'(?<!except)\s', flags=re.IGNORECASE)
    re_r_group = re.compile(r'\(.+?\)')

    blockid = ['BASE', *['IN{:02}'.format(i) for i in range(12)], 'M00', *['OUT{:02}'.format(i) for i in range(12)]]

    def __init__(self, weights: str, instructions: str, verbose: bool = False) -> None:
        cls = PresetInstruction

        self.weights = weights
        self.instructions = instructions
        self.delim = ':'
        self.verbose = verbose
        self.preset_file = 'mbwpresets.txt'

        self.block_data = {}
        weights_list = weights.split(',')
        for i, block in enumerate(cls.blockid):
            self.block_data[block] = {
                'value': float(weights_list[i]),
                'default': float(weights_list[i]),
                'calculate': True if block != 'BASE' else False
            }

    def process(self) -> str:
        cls = PresetInstruction

        instructions = self.instructions.strip()
        instructions = re.sub(r'\s{2,}', ' ', instructions)

        for i, instruction in enumerate(instructions.split(self.delim)):
            if i == 0:
                continue

            instruction = instruction.strip()
            print(f'PresetInstruction: parsed #{i:02} = {instruction}')

            # instruction: calculation
            if (matched := cls.re_i_calculation.match(instruction)) is not None:
                operator = matched.group('operator')
                value = float(matched.group('value'))

                self.__calculate([], [], operator, value)

            # instruction: set
            elif (matched := cls.re_i_set.match(instruction)) is not None:
                params = matched.group('params').strip()

                self.__set(params)

            # instruction: flag operation
            elif (matched := cls.re_i_flag_operation.match(instruction)) is not None:
                keyword = matched.group('keyword')
                params = matched.group('params').strip()

                if keyword == 'set':
                    self.__set(params)
                elif keyword == 'only':
                    self.__only(params)
                elif keyword == 'include':
                    self.__include(params, False, False)
                elif keyword == 'exclude':
                    self.__include(params, True, False)
                elif keyword == 'invert':
                    self.__include(params, False, True)

            # instruction: output
            elif cls.re_i_output.match(instruction) is not None:
                self.__output()

            # instruction: save
            elif (matched := cls.re_i_save.match(instruction)) is not None:
                # params: (PRESET, Overwrite)
                params = matched.group('params').strip()
                params = cls.re_r_whitespace.sub('', params)

                self.__save(params)

            # error
            else:
                print(f'PresetInstruction: invalid instruction {instruction}')

        return ','.join(str(data['value']) for data in self.block_data.values())

    def __calculate(self, included: list[str], excluded: list[str], operator: str, value: float) -> None:
        self.__log(f'PresetInstruction -> calculate: operator = {operator}, value = {value}')
        self.__log(f'PresetInstruction -> calculate: included = {included}, excluded = {excluded}')

        def calculate(fn: Callable[[float, float], float]) -> None:
            for block, data in self.block_data.items():
                if len(included) == 0 and len(excluded) == 0:
                    if data['calculate']:
                        result = fn(data['value'], value)
                        result = max(min(result, 1.5), -1.5)

                        self.block_data[block]['value'] = result
                else:
                    if block in included:
                        if block not in excluded:
                            result = fn(data['value'], value)
                            result = max(min(result, 1.5), -1.5)

                            self.block_data[block]['value'] = result

        if operator == '+':
            calculate(lambda a, b: a + b)
        elif operator == '-':
            calculate(lambda a, b: a - b)
        elif operator == '*':
            calculate(lambda a, b: a * b)
        elif operator == '/':
            calculate(lambda a, b: a / b if b != 0 else a)

    def __only(self, params: str) -> None:
        for block in self.block_data.keys():
            self.block_data[block]['calculate'] = False

        self.__include(params, False, False)

    def __include(self, params: str, is_exclude: bool, is_invert: bool) -> None:
        cls = PresetInstruction

        if is_exclude and is_invert:
            return

        # format parameters
        keyword = ('only/include' if not is_exclude else 'exclude') if not is_invert else 'invert'
        self.__log(f'PresetInstruction -> {keyword}: before = {params}')

        params = cls.re_r_whitespace.sub('', params)
        params = cls.re_r_group.sub(lambda m: m.group(0).replace(',', ';'), params)
        self.__log(f'PresetInstruction -> {keyword}: {params}')

        block_data_before = copy.deepcopy(self.block_data)

        included = []
        excluded = []
        for param in params.split(','):
            in_list, ex_list = self.__targets_to_block_lists(param)
            included = sorted(list(set(included) | set(in_list)), key=lambda x: cls.blockid.index(x))
            excluded = sorted(list(set(excluded) | set(ex_list)), key=lambda x: cls.blockid.index(x))

        self.__log(f'PresetInstruction -> {keyword}: included = {included}')
        self.__log(f'PresetInstruction -> {keyword}: excluded = {excluded}')

        for block in self.block_data.keys():
            if block in included and block not in excluded:
                if not is_invert:
                    self.block_data[block]['calculate'] = True if not is_exclude else False
                else:
                    self.block_data[block]['calculate'] = not self.block_data[block]['calculate']

        changed = []
        for block, data in self.block_data.items():
            if block_data_before[block]['calculate'] != data['calculate']:
                changed.append(block)

        self.__log(f'PresetInstruction -> {keyword}: changed blocks = {changed}')

    def __set(self, params: str) -> None:
        cls = PresetInstruction

        # format parameters
        self.__log(f'PresetInstruction -> set: before = {params}')

        params = cls.re_r_whitespace.sub('', params)
        params = cls.re_r_group.sub(lambda m: m.group(0).replace(',', ';'), params)
        self.__log(f'PresetInstruction -> set: params = {params}')

        for param in params.split(','):
            if (matched := cls.re_c_expression.match(param)) is not None:
                targets = matched.group('targets')
                operator = matched.group('operator')
                value = matched.group('value')

                # <----- left-hand side ----->

                included, excluded = self.__targets_to_block_lists(targets)
                included = sorted(included, key=lambda x: cls.blockid.index(x))
                excluded = sorted(excluded, key=lambda x: cls.blockid.index(x))

                self.__log(f'PresetInstruction -> set: included = {included}')
                self.__log(f'PresetInstruction -> set: excluded = {excluded}')

                # <----- right-hand side ----->

                # value: integer or float
                if (matched := cls.re_v_integer_or_float.match(value)) is not None:
                    value = float(matched.group())
                    value = max(min(value, 1.5), -1.5) if operator == '=' else value
                    self.__log(f'PresetInstruction -> integer or float: value = {value}')

                    if operator == '=':
                        for block in included:
                            if block not in excluded:
                                self.block_data[block]['value'] = value
                    else:
                        # operator = +=, -=, *=, /=
                        self.__calculate(included, excluded, operator[0], value)

                # value: R, U
                elif (matched := cls.re_v_rand_r_or_u.match(value)) is not None:
                    if operator != '=':
                        continue

                    rand_type = matched.group('rand_type').upper()
                    rmin = 0.0 if rand_type == 'R' else -1.5
                    rmax = 1.0 if rand_type == 'R' else 1.5
                    step = 0.001
                    self.__log(f'PresetInstruction -> R, U: rmin = {rmin}, rmax = {rmax}, step = {step}')

                    for block in included:
                        if block not in excluded:
                            self.block_data[block]['value'] = self.get_random(rmin, rmax, step)

                # value: X()
                elif (matched := cls.re_v_rand_x.match(value)) is not None:
                    if operator != '=':
                        continue

                    rmin = float(matched.group('rmin')) if matched.group('rmin') is not None else 0.0
                    rmax = float(matched.group('rmax')) if matched.group('rmax') is not None else 1.0
                    step = float(matched.group('step')) if matched.group('step') is not None else 0.001
                    self.__log(f'PresetInstruction -> X: rmin = {rmin}, rmax = {rmax}, step = {step}')

                    rmin = max(min(rmin, 1.5), -1.5)
                    rmax = max(min(rmax, 1.5), -1.5)
                    step = max(min(step, 3.0), 0.001)

                    for block in included:
                        if block not in excluded:
                            self.block_data[block]['value'] = self.get_random(rmin, rmax, step)

                # error
                else:
                    print(f'PresetInstruction -> set: [ERROR] invalid value {value}')
            else:
                print(f'PresetInstruction -> set: [ERROR] invalid expression {param}')

    def __output(self) -> None:
        output = '\n'
        for block, data in self.block_data.items():
            cvalue = ' ' + str(data['value']) if data['value'] >= 0 else str(data['value'])
            dvalue = ' ' + str(data['default']) if data['default'] >= 0 else str(data['default'])
            output += f'[{block:5}][{"*" if data["calculate"] else " "}] {cvalue:<21} (default = {dvalue:<21})\n'

        print(output)

    def __save(self, params: str) -> bool:
        params_list = params.split(',')

        preset = params_list[0]
        overwrite = params_list[1] if len(params_list) > 1 else ''
        overwrite = overwrite.upper() == 'TRUE'

        self.__log(f'PresetInstruction -> save: preset = {preset}, overwrite = {overwrite}')

        preset_path = os.path.join(basedir(), 'extensions', 'sd-webui-supermerger', 'scripts', self.preset_file)

        if os.path.isfile(preset_path):
            try:
                with open(preset_path, 'r+', encoding='utf-8') as f:
                    lines = f.read().splitlines()
                    lines = list(filter(None, lines))

                    preset_dict = {}
                    for line in lines:
                        if ':' in line:
                            key = line.split(':', 1)[0]
                            preset_dict[key.strip()] = line.split(':', 1)[1]
                        if '\t' in line:
                            key = line.split('\t', 1)[0]
                            preset_dict[key.strip()] = line.split('\t', 1)[1]

                    default_preset_names = [
                        'GRAD_V', 'GRAD_A', 'FLAT_25', 'FLAT_75', 'WRAP08', 'WRAP12', 'WRAP14', 'WRAP16', 'MID12_50',
                        'OUT07', 'OUT12', 'OUT12_5', 'RING08_SOFT', 'RING08_5', 'RING10_5', 'RING10_3', 'SMOOTHSTEP',
                        'REVERSE-SMOOTHSTEP', 'SMOOTHSTEP2', 'R_SMOOTHSTEP2', 'SMOOTHSTEP3', 'R_SMOOTHSTEP3',
                        'SMOOTHSTEP4', 'R_SMOOTHSTEP4', 'SMOOTHSTEP/2', 'R_SMOOTHSTEP/2', 'SMOOTHSTEP/3',
                        'R_SMOOTHSTEP/3', 'SMOOTHSTEP/4', 'R_SMOOTHSTEP/4', 'COSINE', 'REVERSE_COSINE',
                        'TRUE_CUBIC_HERMITE', 'TRUE_REVERSE_CUBIC_HERMITE', 'FAKE_CUBIC_HERMITE',
                        'FAKE_REVERSE_CUBIC_HERMITE', 'ALL_A', 'ALL_B', 'ALL_R', 'ALL_U', 'ALL_X'
                    ]

                    if preset == '' or preset == 'TEMP':
                        preset = 'TEMP'
                        overwrite = True
                        self.__log(f'PresetInstruction -> save: preset = {preset}, overwrite = {overwrite}')

                    if preset in preset_dict.keys() and not overwrite:
                        print(f'PresetInstruction -> save: [ERROR] preset {preset} already exists')
                        return False

                    if preset in default_preset_names:
                        print(f'PresetInstruction -> save: [ERROR] preset {preset} exists in default presets')
                        return False

                    re_preset_name = re.compile(r'^[a-zA-Z0-9_+\-*/]+$')
                    if re_preset_name.match(preset) is None:
                        print(f'PresetInstruction -> save: [ERROR] preset {preset} contains invalid character(s)')
                        return False

                    weights = ','.join(str(data['value']) for data in self.block_data.values())
                    preset_string = f'{preset}\t{weights}'

                    if preset not in preset_dict.keys():
                        lines.append(preset_string)
                        lines.append('')
                    else:
                        line_number = list.index(list(preset_dict.keys()), preset) + 1
                        lines[line_number - 1] = preset_string
                        lines.append('')

                    f.seek(0)
                    content = '\n'.join(lines)
                    f.write(content)
                    f.truncate()

                    self.__log(f'PresetInstruction -> save: {preset_string}')
                    print(f'PresetInstruction -> save: preset {preset} saved to {preset_path}')

            except OSError:
                return False
        else:
            print(f'PresetInstruction -> save: [ERROR] {preset_path} not found')
            return False

        return True

    def __log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def __targets_to_block_lists(self, targets: str, ignore_all: bool = False) -> tuple[list[str], list[str]]:
        cls = PresetInstruction

        included = []
        excluded = []
        for target in targets.split(','):
            # target: all or all(except ...)
            if (matched := cls.re_s_all.match(target)) is not None and not ignore_all:
                params = matched.group('params')
                self.__log(f'targets_to_block_lists -> all: params = {params}')
                if params is not None:
                    for param in params.split(';'):
                        # ignore ALL
                        in_list, _ = self.__targets_to_block_lists(param, True)
                        excluded = sorted(list(set(excluded) | set(in_list)), key=lambda x: cls.blockid.index(x))

                for block in cls.blockid:
                    if block not in excluded and block not in included:
                        included.append(block)

            # target: single block
            elif target in cls.blockid and target not in included:
                self.__log(f'targets_to_block_lists -> single block: target = {target}')
                included.append(target)

            # target: multiple blocks
            elif (matched := cls.re_s_multiple_blocks.match(target)) is not None:
                block_a = matched.group('block_a').upper()
                block_b = matched.group('block_b').upper()
                self.__log(f'targets_to_block_lists -> multiple blocks: block_a = {block_a}, block_b = {block_b}')

                if block_a in cls.blockid and block_b in cls.blockid:
                    block_list = self.__get_blocks_by_range(block_a, block_b)
                    included.extend(block_list)
                else:
                    print(f'targets_to_block_list -> multiple blocks: [ERROR] invalid target {matched.group()}')

            # target: group of blocks like (A, B, C)
            elif (matched := cls.re_s_group.match(target)) is not None:
                targets = matched.group('targets')
                targets = targets.replace(';', ',')
                self.__log(f'targets_to_block_lists -> group: targets = {targets}')

                in_list, ex_list = self.__targets_to_block_lists(targets)
                included = list(set(included) | set(in_list))
                excluded = list(set(excluded) | set(ex_list))

            # target: included or excluded
            elif (matched := cls.re_s_included_or_excluded.match(target)) is not None:
                targets = matched.group('targets').upper()
                self.__log(f'targets_to_block_lists -> included/excluded: target = {targets}')

                for block, data in self.block_data.items():
                    if targets == 'INCLUDED':
                        if data['calculate']:
                            included.append(block)
                    else:
                        if not data['calculate']:
                            included.append(block)

            # target: condition
            elif (matched := cls.re_s_condition.match(target)) is not None:
                operator = matched.group('operator')
                value = float(matched.group('value'))
                self.__log(f'targets_to_block_lists -> condition: operator = {operator}, value = {value}')

                def compare(fn: Callable[[float], bool]) -> None:
                    for compared_block, compared_data in self.block_data.items():
                        if fn(compared_data['value']):
                            included.append(compared_block)

                if operator == '>':
                    compare(lambda v: v > value)
                elif operator == '>=':
                    compare(lambda v: v >= value)
                elif operator == '<':
                    compare(lambda v: v < value)
                elif operator == '<=':
                    compare(lambda v: v <= value)
                elif operator == '==':
                    compare(lambda v: v == value)
                elif operator == '!=':
                    compare(lambda v: v != value)

            # error
            else:
                print(f'PresetInstruction -> targets_to_block_list: [ERROR] invalid target = {target}')

        return included, excluded

    @classmethod
    def __get_blocks_by_range(cls, block_a: str, block_b: str) -> list[str]:
        pos_a = cls.blockid.index(block_a) if block_a in cls.blockid else -1
        pos_b = cls.blockid.index(block_b) if block_b in cls.blockid else -1

        output = []
        if pos_a != -1 and pos_b != -1:
            pos_a, pos_b = cls.swap_if(pos_a, pos_b, lambda a, b: a > b)
            output = cls.blockid[pos_a:pos_b + 1]

        return output

    @staticmethod
    def swap_if(a: float, b: float, fn: Callable[[float, float], bool]) -> tuple[float, float]:
        return (b, a) if fn(a, b) else (a, b)

    @staticmethod
    def get_random(rmin: float, rmax: float, step: float) -> float:
        rmin, rmax = PresetInstruction.swap_if(rmin, rmax, lambda a, b: a > b)
        step = max(step, 0.001)

        rval = np.random.choice(np.arange(rmin, rmax + step if rmax - rmin == 0 else rmax, step))
        rval = min(rval, rmax)

        decimal_places = []
        for number in [rmin, rmax, step]:
            decimal_places.append(len(str(number).split('.')[1]) if '.' in str(number) else 0)

        rval = float(format(rval, f'.{max(decimal_places)}f'))
        return rval
