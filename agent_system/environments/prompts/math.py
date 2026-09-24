MATH_TEMPLATE_NO_HIS = """Solve the math problem step by step.
Problem: {task_description}

At each step, use exactly one action. To calculate, write <python>code</python>.
The interpreter returns <result>output</result>. To finish, write
<answer>\\boxed{{your exact answer}}</answer>.
"""

MATH_TEMPLATE = """Solve the math problem step by step.
Problem: {task_description}

Previous steps:
{memory_context}

Use exactly one action: <python>code</python> or
<answer>\\boxed{{your exact answer}}</answer>.
"""
