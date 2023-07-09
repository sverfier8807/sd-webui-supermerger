# Preset Instructions (experimental)
#
# [set values]
#
# format: set [TARGET] = [VALUE]
#
# [TARGET] = block, blocks (OUT04-OUT11 or reversed order: OUT11-OUT04), ALL, ALL(except block, blocks, all), not implemented: FLAGGED, UNFLAGGED
# [VALUE] = integer or float (within the range from -1.5 to 1.5), and random values: R, U, X()
#
# COSINE : set OUT04=0.25
# COSINE : set OUT04, OUT05=1.00
# COSINE : set BASE, M00, OUT04-OUT11=0.56
# GRAD_V : set ALL=0.45
# GRAD_V : set ALL(except M00, OUT07-OUT11)=1.45
#
# X() takes up to 3 optional arguments, min, max and step. Default values are 0, 1 and 0.001.
#
# ALL_A : set ALL=R
# ALL_A : set ALL=U
# ALL_A : set ALL=X()
# ALL_A : set ALL=X(0.45) -> ALL=X(0.45, 1.00, 0.001)
# ALL_A : set ALL=X(0.45, 0.70) -> ALL=X(0.45, 0.70, 0.001)
# ALL_A : set ALL=X(0.45, 0.70, 0.015) -> ALL=X(0.45, 0.70, 0.015)
#
# multiple instructions:
#
# GRAD_V : set BASE=1.0 : set ALL(except BASE)=R
# GRAD_V : set BASE=U : set ALL(except M00, OUT04)=U : set OUT04=0.55
#
# [calculations]
#
# format: [OPERATOR][VALUE]
#
# SMOOTHSTEP : +0.02
# SMOOTHSTEP : -0.17
# SMOOTHSTEP : *2
# SMOOTHSTEP : /2
#
# ALL_A : +0.25 -> FLAT_25
# ALL_B : /2 -> FLAT_50
# ALL_B : -0.25 -> FLAT_75
#
# BASE is excluded from calculation by default.
# Results are bounded between -1.5 to 1.5.
#
# [set calculation flag]
#
# format: [include or exclude or only] [TARGET]
# format: [include or exclude or only] [OPERATOR][VALUE]
#
# COSINE : /2 (target = all blocks except BASE)
# COSINE : include BASE : /2 (target = all blocks)
# COSINE : include ALL : /2 (target = all blocks)
#
# FAKE_CUBIC_HERMITE : exclude ALL : /2 (target = none)
# FAKE_CUBIC_HERMITE : exclude M00, OUT04-OUT07 : /2 (target = all blocks except BASE, M00, OUT04 to OUT07)
#
# FAKE_CUBIC_HERMITE : only OUT04-OUT05 : +0.15 (target = OUT04 and OUT05)
# FAKE_CUBIC_HERMITE : only ALL : +0.15 (target = all blocks)

# with operators (>, >=, <=, <, !=, ==):
#
# COSINE : include > 0.50 : /2 (target = all blocks match the condition: > 0.50)
# COSINE : exclude > 0.50 : /2
# COSINE : only < 0.56 : +0.05
#
# [not implemented]
#
# FLAGGED and UNFLAGGED keywords, operators in ALL(except ...)

import re
import numpy as np

blockid = ['BASE', *['IN{:02}'.format(i) for i in range(12)], 'M00', *['OUT{:02}'.format(i) for i in range(12)]]

def process_instructions(weights, instructions, delim):
    re_calculation = re.compile(r"""
    ^
        (?P<operator>[+\-*/]) # operator
        \s?
        (?P<value>\d+\.\d*|\d*\.\d+|\d+) # integer or float
    $
    """, re.VERBOSE)

    re_keywords = re.compile(r"""
    ^
        (?P<keyword>set|only|include|exclude) # keyword: set, only, include, exclude
        \s
        (?P<params>(.+)) # parameters
    $
    """, re.IGNORECASE | re.VERBOSE)

    re_compare = re.compile(r"""
    ^
        (?P<operator>>|>=|<|<=|==|!=) # operator: >, >=, <, <=, ==, !=
        \s?
        (?P<value>\d+\.\d*|\d*\.\d+|\d+) # integer or float
    $
    """, re.VERBOSE)

    re_rand_r_or_u = re.compile(r"""
    ^
        (?P<rand_type>r|u)
    $
    """, re.IGNORECASE | re.VERBOSE)
    
    re_rand_x = re.compile(r"""
    ^
        x\(
            (?P<rmin>\d+\.\d*|\d*\.\d+|\d+)? # integer or float
            (?:;(?P<rmax>\d+\.\d*|\d*\.\d+|\d+))? # integer or float
            (?:;(?P<step>\d+\.\d*|\d*\.\d+|\d+))? # integer or float
        \)
    $
    """, re.IGNORECASE | re.VERBOSE)

    re_all = re.compile(r'^all(?:\(except\s(?P<except_params>.+)\))?$', re.IGNORECASE)
    re_value = re.compile(r'^(?P<value>\d+\.\d*|\d*\.\d+|\d+)$')
    re_expression = re.compile(r'^(?P<left>(.+))\s?=\s?(?P<right>(.+))$')
    re_sub_whitespace = re.compile(r'(?<!except)\s', flags=re.IGNORECASE)
    re_sub_rand = re.compile(r'x\((.*?)\)', flags=re.IGNORECASE)
    re_sub_all = re.compile(r'all(?:\(except (.*)\))?', flags=re.IGNORECASE) 
    # re_sub_rand = re.compile(r'rand\((.*?)\)', flags=re.IGNORECASE)

    # create dictionary: {'BLOCK': {'default': integer or float, 'value': integer or float, 'bit': boolean}, ...}
    block_data = {}
    weights_list = weights.split(',')
    for i, elem in enumerate(blockid):
        block_data[elem] = {
            'default': float(weights_list[i]),
            'value': float(weights_list[i]),
            'bit': True if elem != 'BASE' else False
        }

    def swap_if(a, b, fn):
        return (b, a) if fn(a, b) else (a, b)

    def get_max_decimal_place(*numbers):
        places = []
        for number in numbers:
            places.append(len(str(number).split('.')[1]) if '.' in str(number) else 0)

        return max(places)

    def get_random(rmin, rmax, step):
        rmin, rmax = swap_if(rmin, rmax, lambda a, b: a > b)

        decimal_place = get_max_decimal_place(rmin, rmax)
        step = 1 / (10 ** decimal_place) if int((rmax - rmin) / step) == 0 else step

        rval = np.random.choice(np.arange(rmin, rmax, step))

        decimal_place = get_max_decimal_place(rmin, rmax, step)
        rval = float(format(rval, f'.{decimal_place}f'))
        return rval

    def get_blocks_by_range(block_a, block_b):
        pos_a = blockid.index(block_a) if block_a in blockid else -1
        pos_b = blockid.index(block_b) if block_b in blockid else -1

        output = []
        if pos_a != -1 and pos_b != -1:
            pos_a, pos_b = swap_if(pos_a, pos_b, lambda a, b: a > b)
            output = blockid[pos_a:pos_b + 1]

        return output

    def process_calculation(operator, value):
        print(f'process_instruction: type: calculation, operator = {operator}, value = {value}')

        def calculate(fn):
            nonlocal block_data
            for key, data in block_data.items():
                if data['bit']:
                    result = fn(data['value'], value)
                    result = max(min(result, 1.5), -1.5)

                    # exponential
                    if 'e' in str(result).lower():
                        print('process_calculation: WARNING: exponential notation found, set to 0')
                        result = 0.0

                    block_data[key]['value'] = result

        if operator == '+':
            calculate(lambda a, b: a + b)
        elif operator == '-':
            calculate(lambda a, b: a - b)
        elif operator == '*':
            calculate(lambda a, b: a * b)
        elif operator == '/':
            calculate(lambda a, b: a / b if b != 0 else a)

    def process_set(params):
        nonlocal block_data
        params = re_sub_whitespace.sub('', params)
        params = re_sub_all.sub(lambda m: m.group(0).replace(',', ';'), params)
        params = re_sub_rand.sub(lambda m: m.group(0).replace(',', ';'), params)

        print(f'process_set: params = {params}')

        for param in params.split(','):
            if (matched := re_expression.match(param)) is not None:
                left = matched.group('left')
                right = matched.group('right')

                inclusions = []
                exclusions = []

                # single block: [BLOCK]
                if left in blockid and left not in inclusions:
                    inclusions.append(left)

                # all: all or all(except [BLOCK] or [BLOCK]-[BLOCK], ...)
                elif (matched := re_all.match(left)) is not None:
                    except_params = matched.group('except_params')
                    if except_params is not None:
                        for exparam in except_params.split(';'):
                            # single block: [BLOCK]
                            if exparam in blockid:
                                exclusions.append(exparam)

                            # multiple blocks: [BLOCK]-[BLOCK]
                            elif (blocks := exparam.split('-')) and len(blocks) == 2:
                                block_list = get_blocks_by_range(blocks[0], blocks[1])
                                for block in block_list:
                                    if block not in exclusions:
                                        exclusions.append(block)

                                print(f'process_set: exclusions = {exclusions}')

                    for key in block_data.keys():
                        if key not in exclusions and key not in inclusions:
                            inclusions.append(key)

                # multiple blocks: [BLOCK]-[BLOCK]
                elif (blocks := left.split('-')) and len(blocks) == 2:
                    block_list = get_blocks_by_range(blocks[0], blocks[1])
                    inclusions.extend(block_list)
                else:
                    print(f'process_set: {left} not found or invalid format for ALL/ALL(except...)')

                # [VALUE] integer or float
                if re_value.match(right) is not None:
                    for block in inclusions:
                        if block not in exclusions:
                            block_data[block]['value'] = max(min(float(right), 1.5), -1.5)

                # [R or U] random
                elif (matched := re_rand_r_or_u.match(right)) is not None:
                    rand_type = matched.group('rand_type')

                    rmin = 0.0 if rand_type == 'R' or rand_type == 'r' else -1.5
                    rmax = 1.0 if rand_type == 'R' or rand_type == 'r' else 1.5
                    step = 0.001

                    for block in inclusions:
                        if block not in exclusions:
                            block_data[block]['value'] = get_random(rmin, rmax, step)
                
                # [X()] random
                elif (matched := re_rand_x.match(right)) is not None:
                    rmin = float(matched.group('rmin')) if matched.group('rmin') is not None else 0.0
                    rmax = float(matched.group('rmax')) if matched.group('rmax') is not None else 1.0
                    step = float(matched.group('step')) if matched.group('step') is not None else 0.001

                    rmin = max(rmin, -1.5)
                    rmax = min(rmax, 1.5)
                    step = max(step, 0.001)
                    
                    for block in inclusions:
                        if block not in exclusions:
                            block_data[block]['value'] = get_random(rmin, rmax, step)

                else:
                    print(f'process_set: {right} is not value or invalid format for R, U and X')

    def process_only(params):
        nonlocal block_data
        for key in block_data.keys():
            block_data[key]['bit'] = False

        print(f'process_only: ALL set to False')
        process_include(params)

    def process_include(params, flip_result=False):
        nonlocal block_data
        params = re_sub_whitespace.sub('', params)
        params = re_sub_all.sub(lambda m: m.group(0).replace(',', ';'), params)

        print(f'process_include: params = {params}, flip_result = {flip_result}')

        for param in params.split(','):
            # single block: [BLOCK]
            if param in blockid:
                block_data[param]['bit'] = not flip_result
                print(f'process_include: {param} set to {not flip_result}')

            # all: all or all(except [BLOCK] or [BLOCK]-[BLOCK], ...)
            elif (matched := re_all.match(param)) is not None:
                exclusions = []
                exparams = matched.group('except_params')
                if exparams is not None:
                    for exparam in exparams.split(';'):
                        # single block: [BLOCK]
                        if exparam in blockid and exparam not in exclusions:
                            exclusions.append(exparam)

                        # multiple blocks: [BLOCK]-[BLOCK]
                        elif (blocks := exparam.split('-')) and len(blocks) == 2:
                            if blocks[0] in blockid and blocks[1] in blockid:
                                block_list = get_blocks_by_range(blocks[0], blocks[1])
                                for block in block_list:
                                    if block not in exclusions:
                                        exclusions.append(block)

                    print(f'process_include: exclusions = {exclusions}')

                changed = []
                for key in block_data.keys():
                    if key not in exclusions:
                        changed.append(key)
                        block_data[key]['bit'] = not flip_result

                print(f'process_include: changed = [{changed}] to {not flip_result}')

            # multiple blocks: [BLOCK]-[BLOCK]
            elif (blocks := param.split('-')) and len(blocks) == 2:
                if blocks[0] in blockid and blocks[1] in blockid:
                    block_list = get_blocks_by_range(blocks[0], blocks[1])
                    for block in block_list:
                        block_data[block]['bit'] = not flip_result

                    print(f'process_include: [{block_list}] set to {not flip_result}')

            # compare: include [OPERATOR: >, >=, <, <=, ==, !=] [VALUE]
            elif (matched := re_compare.match(param)) is not None:
                operator = matched.group('operator')
                value = float(matched.group('value'))

                def compare(fn):
                    nonlocal block_data
                    for k, d in block_data.items():
                        block_data[k]['bit'] = fn(d['value']) if not flip_result else not fn(d['value'])

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

    def process_exclude(params):
        process_include(params, True)

    # remove redundant whitespaces and tabs
    instructions = instructions.strip()
    instructions = re.sub(r'\s{2,}', ' ', instructions)

    for i, instruction in enumerate(instructions.split(delim)):
        if i == 0:
            continue

        instruction = instruction.strip()

        print(f'process_instruction: parsed #{i:02} = {instruction}')

        # calculation
        if (matched_instruction := re_calculation.match(instruction)) is not None:
            arg1 = matched_instruction.group('operator')
            arg2 = float(matched_instruction.group('value'))

            process_calculation(arg1, arg2)

        # set, only, include, exclude
        elif (matched_instruction := re_keywords.match(instruction)) is not None:
            keyword = matched_instruction.group('keyword')
            arg1 = matched_instruction.group('params').strip()

            if keyword == 'set':
                process_set(arg1)
            elif keyword == 'only':
                process_only(arg1)
            elif keyword == 'include':
                process_include(arg1)
            elif keyword == 'exclude':
                process_exclude(arg1)

        else:
            print(f'process_instruction: [ERROR] invalid format "{instruction}"')

    return ','.join(str(data['value']) for data in block_data.values())