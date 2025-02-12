# Python
---
## Resources
- Available on [The Python Tutorial](https://docs.python.org/3/tutorial/).
- Available on [The Python Standard Library](https://docs.python.org/3/library/index.html).
- Available on [RUNOOB.COM](https://www.runoob.com/python/python-tutorial.html). (Suitable for Chinese speaker)

---
### Python Operators
#### **<u>Arithmetic Operators</u>**
- `+`：加法
- `-`：减法
- `*`：乘法
- `/`：除法
- `//`：整除（返回商的整数部分）
- `%`：取模（返回除法的余数）
- `**`：幂运算（求 x 的 y 次幂）

#### **<u>Assignment Operators</u>**
- `=`：简单的赋值运算符
- `+=`：加法赋值运算符（`a += b` 相当于 `a = a + b`）
- `-=`：减法赋值运算符
- `*=`：乘法赋值运算符
- `/=`：除法赋值运算符
- `//=`：整除赋值运算符
- `%=`：取模赋值运算符
- `**=`：幂赋值运算符
- `&=`：按位与赋值
- `|=`：按位或赋值
- `^=`：按位异或赋值
- `>>=`：右移赋值
- `<<=`：左移赋值
- `:=`：海象运算符，用于在表达式中进行赋值   # `x := 3` 意为将值 `3` 赋给变量 `x`

#### **<u>Comparison Operators</u>**
这些运算符用于比较两个值，并返回布尔值（`True` 或 `False`）。
- `==`：等于
- `!=`：不等于
- `>`：大于
- `<`：小于
- `>=`：大于等于
- `<=`：小于等于

#### **<u>Logical Operators</u>**
这些运算符用于组合多个布尔表达式
- `and`：逻辑与（只有当所有条件都为 True 时，结果才为 True）
- `or`：逻辑或（只要有一个条件为 True，结果就为 True）
- `not`：逻辑非（反转布尔值）

#### **<u>Identity Operators</u>**
这些运算符用于比较两个对象的身份（是否为同一个对象）。
- `is`：检查两个变量是否引用同一个对象
- `is not`：检查两个变量是否不引用同一个对象

#### **<u>Membership Operators</u>**
这些运算符用于检查某个值是否存在于序列（如列表、元组、字符串）中。
- `in`：如果值在序列中，返回 `True`
- `not in`：如果值不在序列中，返回 `True`
e.g. 
  ```
  a = [1, 2, 3]
  b = [1, 2, 3]
  c = a
  # 检查 a 和 b 是否引用同一个对象
  result_is = a is b  # 结果为 False，因为 a 和 b 是两个不同的列表对象
  # 检查 a 和 c 是否引用同一个对象
  result_is_not = a is not c  # 结果为 False，因为 a 和 c 引用同一个对象
  ```

#### **<u>Bitwise Operators</u>**
这些运算符用于执行位级操作，即对整数的二进制位进行操作。
按位运算是指对二进制数的每一位进行操作的运算
- `&`：按位与
- `|`：按位或
- `^`：按位异或
- `~`：按位取反
- `<<`：左移
- `>>`：右移
e.g.
  ```
  a = 5          # 5的二进制表示为 101
  b = 3          # 3的二进制表示为 011
  
  # 按位与：只有当两个位都为 1 时，结果位才为 1
  result_and = a & b  # 101 & 011 = 001，即 1
  
  # 按位或：只要有一个位为 1，结果位就为 1
  result_or = a | b   # 101 | 011 = 111，即 7
  
  # 按位异或：只有当两个位不同时，结果位才为 1
  result_xor = a ^ b  # 101 ^ 011 = 110，即 6
  
  # 按位取反：反转所有位
  result_not = ~a     # ~101 = 0100，即 -6（注意：Python 使用补码表示负数）
  
  # 左移：将二进制位向左移动，右边补 0
  result_left_shift = a << 1  # 101 << 1 = 1010，即 10
  
  # 右移：将二进制位向右移动，左边补符号位（0 或 1）
  result_right_shift = a >> 1  # 101 >> 1 = 110，即 2
  ```

##### **Operator Precedence**
The precedence order is described in the table below, starting with the highest precedence at the top
| Operator                                                | Description                                           |
| ------------------------------------------------------- | ----------------------------------------------------- |
| `()`                                                    | Parentheses                                           |
| `**`                                                    | Exponentiation                                        |
| `+x` `-x` `~x`                                          | Unary plus, unary minus, and bitwise NOT              |
| `*` `/` `//` `%`                                        | Multiplication, division, floor division, and modulus |
| `+` `-`                                                 | Addition and subtraction                              |
| `<<` `>>`                                               | Bitwise left and right shifts                         |
| `&`                                                     | Bitwise AND                                           |
| `^`                                                     | Bitwise XOR                                           |
| `                                                       | `                                                     | Bitwise OR |
| `==` `!=` `>` `>=` `<` `<=` `is` `is not` `in` `not in` | Comparisons, identity, and membership operators       |
| `not`                                                   | Logical NOT                                           |
| `and`                                                   | AND                                                   |
| `or`                                                    | OR                                                    |

If two operators have the same precedence, the expression is evaluated from left to right.

---
### List
**<u>List Items</u>**
- List items are ordered, changeable, and allow duplicate values.
- List items are indexed, the first item has index [0], the second item has index [1] etc.
  
**<u>Ordered</u>**
- the items have a defined order, and that order will not change.
- If add new items to a list, the new items will be placed at the end of the list

**<u>Changeable</u>**
- we can change, add, and remove items in a list after it has been created

**<u>Allow Duplicates</u>**
- Since lists are indexed, lists can have items with the same value

#### **<u>List methods</u>**
| Method      | Description                                                                  |
| ----------- | ---------------------------------------------------------------------------- |
| `append()`  | Adds an element at the end of the list                                       |
| `clear()`   | Removes all the elements from the list                                       |
| `copy()`    | Returns a copy of the list                                                   |
| `count()`   | Returns the number of elements with the specified value                      |
| `extend()`  | Add the elements of a list (or any iterable), to the end of the current list |
| `index()`   | Returns the index of the first element with the specified value              |
| `insert()`  | Adds an element at the specified position                                    |
| `pop()`     | Removes the element at the specified position                                |
| `remove()`  | Removes the item with the specified value                                    |
| `reverse()` | Reverses the order of the list                                               |
| `sort()`    | Sorts the list                                                               |

**Note:** 
- The `index()` method only returns the first occurrence of the value.
- The `pop()` method returns removed value
  ```
  fruits = ['apple', 'banana', 'cherry']
  fruits.pop(1)
  print(fruits)

  #输出结果为['apple', 'cherry']
  ```
  ```
  fruits = ['apple', 'banana', 'cherry']
  x = fruits.pop(1)
  print(x)

  #输出结果为banana
  ```
  
  