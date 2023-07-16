## Preset Instuction (experimental)
You can modify the preset weights without making changes to the preset file.
<br>
This feature is available for **mbw alpha**, **mbw beta**, and **mbw alpha and mbw beta** modes in X/Y/Z plot.
<br><br>
All instructions including block names are not case-senstive.

### calculation
Each block has a calculation flag that determines whether it is included or excluded from calculations.
<br>
**BASE** is excluded from calculations by default.
<br><br>
Default calculation flags:

| BASE  | IN00 - OUT11 |
|-------|--------------|
| False | True         |

#### syntax: \[operator\] \[value\]
*operator: +, -, \*, /*
<br>
*value: integer or float (incomplete float values are also valid: 0. or .0 )*

***examples***

| Instructions   | 
|----------------|
| COSINE : +0.15 |
| COSINE : -0.15 |
| COSINE : *2    |
| COSINE : /2    |

Above instructions perform a calculation excluding **BASE**.
<br>
You can also perform calculations on specific blocks with using the **set** instruction.

### block selection
There are four instructions to select blocks for calculations.
<br><br>
**only**: Set the calculation flag to True for the specified blocks and **set False to other blocks**.
<br>
**include**: Set the calculation flag to True for the specified blocks.
<br>
**exclude**: Set the calculation flag to False for the specified blocks.
<br>
**invert**: Invert flags of the specified blocks.

#### syntax: \[instruction\] \[block|cond\], ...

*instruction: only, include, exclude, invert*
<br>
*block: single block, multiple blocks, 'ALL', 'ALL(except ...)', 'INCLUDED', 'EXCLUDED'*

***examples***

| Instructions                     | Description                                             |
|----------------------------------|---------------------------------------------------------|
| COSINE : only OUT04 : /2         | Perform /2 only on OUT04.                               |
| COSINE : include BASE : +0.15    | Perform +0.15 on all blocks.                            |
| COSINE : exclude M00, OUT04 : /2 | Perform /2 on all blocks excluding BASE, M00 and OUT04. |
| COSINE : invert ALL : /2         | Perform /2 on BASE.                                     |


You can use **INCLUDED** and **EXCLUDED** as specifiers.
<br>
**INCLUDED**: All blocks that are included in calculations.
<br>
**EXCLUDED**: All blocks that are excluded from calculations.

***examples***

| Instruction                            | Description                          |
|----------------------------------------|--------------------------------------|
| R_SMOOTHSTEP*2 : include INCLUDED : /2 | No changes to the calculation flags. |
| R_SMOOTHSTEP*2 : include EXCLUDED : /2 | Equivalent to **include ALL**.       |
| R_SMOOTHSTEP*2 : exclude INCLUDED : /2 | Equivalent to **exclude ALL**.       |
| R_SMOOTHSTEP*2 : exclude EXCLUDED : /2 | No changes to the calculation flags. |

#### syntax (multiple blocks): \[block\]-\[block\]
*block: single block*

***examples***

| Instruction                               | Description                                                    |
|-------------------------------------------|----------------------------------------------------------------|
| REVERSE_COSINE : exclude OUT04-OUT07 : /2 | Exclude OUT04 to OUT07 then perform /2 on all included blocks. | 
| REVERSE_COSINE : exclude OUT07-OUT04 : /2 | Reversed order is also valid: **OUT07-OUT04**                  |

#### syntax (ALL): ALL(except \[block\], ...)
*block: single block, multiple blocks, 'INCLUDED', 'EXCLUDED'*

***examples***

| Instruction                                              | Description               |
|----------------------------------------------------------|---------------------------|
| COSINE : include ALL : /2                                | Perform /2 on all blocks. |
| COSINE : exclude ALL(except M00) : /2                    | Perform /2 on M00.        |
| COSINE : only ALL(except BASE, M00, OUT04-OUT07) : +0.15 | Perform /2 on all blocks  |

#### syntax (cond): \[operator\] \[value\]
*operator: >, >=, <=, <, ==, !=*
<br>
*value: integer or float*

***examples***

| Instruction                  | Description                                                         |
|------------------------------|---------------------------------------------------------------------|
| GRAD_V : only > 0.5 : *2     | Perform *2 on all blocks whose weight is greater than 0.5.          |
| GRAD_V : only < 0.5 : *2     | Perform *2 on all blocks whose weight is less than 0.5              |
| GRAD_V : exclude == 0.5 : *2 | Exclude all blocks whose weight is equal to 0.5 then perform *2.    |
| GRAD_V : invert != 0.5 : *2  | Invert all blocks whose weight is not equal to 0.5 then perform *2. |

### set value / calculation per block

#### syntax (set value): set \[block\] = \[value|R|U|X()\]

*block: single block, multiple block, group of blocks, 'ALL', 'ALL(except ...)'*
<br>
*value: integer or float*
<br>
*R: replaced by a random value within the range from 0 to 1*
<br>
*U: replaced by a random value within the range from -1.5 to 1.5*
<br>
*X(): replaced by a random value within the range from min to max with step*
<br>
*X(): takes up to 3 optional arguments: min, max and step, **default values are 0, 1, and 0.001***

#### syntax (group of blocks): (\[block|cond\], ...)
*block: single block, multiple bocks, cond*

***examples***

| Instruction                               | Description                                                   |
|-------------------------------------------|---------------------------------------------------------------|
| SMOOTHSTEP : set OUT04 = 0.25             | Set 0.25 to OUT04.                                            |
| SMOOTHSTEP : set **M00**, OUT04 = 0.25    | Set 0.25 to OUT04. **M00 is ignored.** Use () to include M00. |
| SMOOTHSTEP : set (M00, OUT04) = 0.25 : /2 | Set 0.25 to M00 and OUT04.                                    |

***examples (condition)***

| Instruction                    | Description                                               |
|--------------------------------|-----------------------------------------------------------|
| SMOOTHSTEP : set > 0.30 = 0.25 | Set 0.25 to all blocks whose weight is greater than 0.30. |
| SMOOTHSTEP : set < 0.30 = 0.15 | Set 0.15 to all blocks whose weight is less than 0.30.    |
| SMOOTHSTEP : set == 1 = 0.25   | Set 0.25 to all blocks whose weight is equal to 1.00.     |
| SMOOTHSTEP : set != 1 = 0.15   | Set 0.15 to all blocks whose weight is not equal to 1.00. |

***examples (random values)***

| Instruction                                       | Random Value                             |
|---------------------------------------------------|------------------------------------------|
| SMOOTHSTEP : set OUT04 = R                        | Range: from 0.00 to 1.00 step 0.001.     |
| SMOOTHSTEP : set OUT04 = U                        | Range: from -1.50 to 1.50 step 0.001.    |
| SMOOTHSTEP : set OUT04 = X()                      | Range: from 0.00 to 1.00 step 0.001.     |
| SMOOTHSTEP : set OUT04 = X(0.25)                  | Range: from 0.25 to 1.00 step 0.001.     |
| SMOOTHSTEP : set OUT04 = X(0.25, 0.50)            | Range: from 0.25 to 0.50 step 0.001.     |
| SMOOTHSTEP : set OUT04 = X(0.25, 0.50, **0.015**) | Range: from 0.25 to 0.50 step **0.015**. |

#### syntax (calculation): set \[block\] \[operator\] \[value\]

**<span style="color: blue">This instruction ignores calculation flags.</span>**

*block: single block, multiple block, group of blocks, 'ALL', 'ALL(except ...)'*
<br>
*operator: +=, -=, \*=, /=*
<br>
*value: integer or float*

***examples***

| Instruction                                 | Description                                  |
|---------------------------------------------|----------------------------------------------|
| COSINE : set M00 += 0.25                    | Perform +0.25 on M00.                        |
| COSINE : set (IN00, IN01) -= 0.25           | Perform -0.25 on IN00 and IN01.              |
| COSINE : set (IN00, IN04, OUT04-OUT07) *= 2 | Perform *2 on IN00, IN04 and OUT04 to OUT07. |
| COSINE : set (IN00, IN04, OUT04-OUT07) /= 2 | Perform /2 on IN00, IN04 and OUT04 to OUT07. |

### output

Output current weights and calculation flags.

#### syntax: output

***examples***

| Instruction                     | Description                                                      |
|---------------------------------|------------------------------------------------------------------|
| COSINE : set OUT04 = R : output | Set random value to OUT04 then output current weights and flags. |

### save preset

Save current weights as a new preset or update an existing preset. Default presets cannot be overwritten.
<br>
If no preset name is specified, it is saved as TEMP.
<br>
**The TEMP preset will always be overwritten regardless of the overwrite flag.**

#### syntax: save(preset, overwrite)

**<span style="color: blue">This instruction does not reload presets.</span>**

***examples***

| Instruction                                           | Description                                           |
|-------------------------------------------------------|-------------------------------------------------------|
| COSINE : set (OUT04, OUT05) = U : save()              | Save as TEMP or overwrite TEMP.                       |
| COSINE : set (OUT04, OUT05) = U : save(COSINE2)       | Save as COSINE2, will fail if COSINE2 already exists. |
| COSINE : set (OUT04, OUT05) = U : save(COSINE2, True) | Save as COSINE2 or overwrite COSINE2.                 |
