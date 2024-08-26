## 1. Input Data

### Example Input

```json
{
  "name": "abc",
  "score": 99,
  "date": "01/02/2025",
  "animal": "dog",
  "flag": true
}
```

# 1. Encoding

String ("abc")

```text
"abc" -> [1, 2, 3, 0]
# Mapped indices: 'a'->1, 'b'->2, 'c'->3, padding->0
```

Integer (99)

````text
99 -> [99.0]

Date ("01/02/2025")
```text
"01/02/2025" -> [0.693, 0.0909, 0.0323]
# Normalized year, month, day
````

Category ("dog")

```text
"dog" -> [1.0]
# Assuming categories: ['cat', 'dog', 'bird']
```

Boolean (True)

```text
True -> [1.0]
```

# 2. Embedding Layer

```text
+------------------------------------------------------------------+
|                          Embedding Layer                         |
|                                                                  |
|  +-------+    +-------+    +-------+    +-------+    +-------+   |
|  | Input | -> | Embed | -> | Embed | -> | Embed | -> | Embed |   |
|  +-------+    +-------+    +-------+    +-------+    +-------+   |
|                                                                  |
|  "abc" -> [1, 2, 3, 0]     -> [[e1], [e2], [e3], [e0]]           |
|  99     -> [99.0]          -> [[e99]]                            |
|  Date   -> [0.693, 0.0909, 0.0323] -> [[ed1], [ed2], [ed3]]      |
|  "dog"  -> [1.0]           -> [[edog]]                           |
|  True   -> [1.0]           -> [[etrue]]                          |
|                                                                  |
|  Embedding Dimension (d_model) = 4                               |
+------------------------------------------------------------------+
```

Example Embeddding outputs:

```text
"abc" -> [[0.5, 0.1, -0.2, 0.3], [0.4, 0.3, 0.1, -0.1], [0.2, 0.2, 0.5, 0.4], [0.0, 0.0, 0.0, 0.0]]
99 -> [[0.7, 0.6, 0.8, 0.9]]
Date -> [[0.4, 0.5, 0.3, 0.2], [0.1, 0.1, 0.2, 0.3], [0.3, 0.2, 0.4, 0.6]]
"dog" -> [[0.9, 0.8, 0.7, 0.6]]
True -> [[1.0, 0.9, 0.8, 0.7]]
```

# 3. Positional Encoding

```
+----------------------------------------------+
|               Positional Encoding            |
|                                              |
|  Embedding -> Positionally Encoded Embedding |
|                                              |
|  "abc" -> [[0.5+p1, 0.1+p1, -0.2+p1, 0.3+p1],|
|            [0.4+p2, 0.3+p2, 0.1+p2, -0.1+p2],|
|            [0.2+p3, 0.2+p3, 0.5+p3, 0.4+p3],|
|            [0.0+p4, 0.0+p4, 0.0+p4, 0.0+p4]]|
|                                              |
+----------------------------------------------+

```

# 4. Transformer

Self-Attention

```text
+-----------------------------------------+
|              Self-Attention             |
|                                         |
|  Inputs interact and attend to each     |
|  other, capturing dependencies.         |
|                                         |
|  "abc" + 99 + Date + "dog" + True       |
|  -----------------------------------    |
|  Key:Value pairs calculated, weights    |
|  derived, weighted sums computed.       |
+-----------------------------------------+
```

Multi-head Attention

```text
+-------------------------------------------------------+
|                  Multi-Head Attention                 |
|                                                       |
|  Heads:                                               |
|                                                       |
|  Head 1: Looks at relationships across all elements   |
|  Head 2: Focuses on different aspects of relationships|
|  Head 3: Captures yet another view of dependencies    |
|  Head 4: Different focus on relationships             |
|                                                       |
|  Each head looks at every element in the sequence,    |
|  but in different ways, capturing diverse information.|
+-------------------------------------------------------+
```

# 5. Output Layers

String

```text
+--------------------------------------------------------+
|                    String Output Layer                 |
|                                                        |
|  Input: [batch_size, max_len, d_model]                 |
|                                                        |
|  Linear Layer                                          |
|  (Projects d_model -> total_characters * max_len)      |
|                                                        |
|  Output Shape: [batch_size, max_len, total_characters] |
|                                                        |
|  Softmax (Per Character)                               |
|                                                        |
|  Argmax -> Predicted Indices                           |
|                                                        |
|  Decoding (Indices -> Characters)                      |
|                                                        |
|  Output: "predicted_string"                            |
+--------------------------------------------------------+
```

Integer

```text
+------------------------------------------+
|              Integer Output Layer        |
|                                          |
|  Input: [batch_size, d_model]            |
|                                          |
|  Linear Layer                            |
|  (Projects d_model -> 1)                 |
|                                          |
|  Output Shape: [batch_size, 1]           |
|                                          |
|  Scaling (If Necessary)                  |
|                                          |
|  Output: Predicted Integer               |
+------------------------------------------+
```

Float

```text
+------------------------------------------+
|              Float Output Layer          |
|                                          |
|  Input: [batch_size, d_model]            |
|                                          |
|  Linear Layer                            |
|  (Projects d_model -> 1)                 |
|                                          |
|  Output Shape: [batch_size, 1]           |
|                                          |
|  Scaling (If Necessary)                  |
|                                          |
|  Output: Predicted Float                 |
+------------------------------------------+
```

Date

```text
+-----------------------------------------------------+
|                   Date Output Layer                 |
|                                                     |
|  Input: [batch_size, d_model]                       |
|                                                     |
|  Linear Layer                                       |
|  (Projects d_model -> 3)                            |
|                                                     |
|  Output Shape: [batch_size, 3]                      |
|  (Year, Month, Day)                                 |
|                                                     |
|  Denormalization                                    |
|  Year -> Reverse Logarithmic Transformation         |
|  Month -> Scale to [1, 12]                          |
|  Day -> Scale to [1, 31]                            |
|                                                     |
|  Output: Predicted Date (e.g., "01/03/2025")        |
+-----------------------------------------------------+
```

Category

```text
+---------------------------------------------+
|            Category Output Layer            |
|                                             |
|  Input: [batch_size, d_model]               |
|                                             |
|  Linear Layer                               |
|  (Projects d_model -> num_categories)       |
|                                             |
|  Output Shape: [batch_size, num_categories] |
|                                             |
|  Softmax -> Probability Distribution        |
|                                             |
|  Argmax -> Predicted Category               |
|                                             |
|  Output: "predicted_category"               |
+------------------------------------------+

```

Boolean

```text
+------------------------------------------+
|              Boolean Output Layer        |
|                                          |
|  Input: [batch_size, d_model]            |
|                                          |
|  Linear Layer                            |
|  (Projects d_model -> 2)                 |
|  (One output for True, one for False)    |
|                                          |
|  Output Shape: [batch_size, 2]           |
|                                          |
|  Softmax -> Probability Distribution     |
|                                          |
|  Argmax -> Predicted Boolean             |
|                                          |
|  Output: True/False                      |
+------------------------------------------+
```
