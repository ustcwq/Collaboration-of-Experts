from __future__ import annotations

import unittest

from bench_coe.innovation.bbh_symbolic import (
    solve_boolean_expression,
    solve_dyck_completion,
    solve_multistep_arithmetic,
    solve_word_sorting,
)


class BBHSymbolicTests(unittest.TestCase):
    def test_boolean_expression(self) -> None:
        self.assertEqual(solve_boolean_expression("not ( True ) and ( True ) is"), "False")
        self.assertIsNone(solve_boolean_expression("__import__('os').system('id') is"))

    def test_multistep_arithmetic(self) -> None:
        self.assertEqual(
            solve_multistep_arithmetic("((-1 + 2 + 9 * 5) - (-2 + -4 + -4 * -7)) ="),
            "24",
        )
        self.assertIsNone(solve_multistep_arithmetic("2 ** 10 ="))

    def test_dyck_completion(self) -> None:
        self.assertEqual(
            solve_dyck_completion(
                "Complete the rest of the sequence. Input: < [ ["
            ),
            "] ] >",
        )
        self.assertIsNone(solve_dyck_completion("Input: [ >"))

    def test_word_sorting(self) -> None:
        self.assertEqual(
            solve_word_sorting("Sort alphabetically. List: zebra apple apple"),
            "apple apple zebra",
        )


if __name__ == "__main__":
    unittest.main()
