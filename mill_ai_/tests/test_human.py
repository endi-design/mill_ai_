"""Test mouse-action filtering with a fake interaction wrapper (no GUI)."""

import unittest

from human_play import HumanQuit, choose_human_action


class FakeUI:
    def __init__(self, clicks):
        self.clicks = iter(clicks)
        self.marked = []

    def clear_markings(self):
        self.marked.clear()

    def mark_position(self, pos, color):
        self.marked.append(pos)

    def interact(self):
        return next(self.clicks)


def click(pos):
    return {"type": "mouse_click", "position": pos}


class TestHumanInput(unittest.TestCase):
    def test_placing_piece(self):
        ui = FakeUI([click(5)])
        self.assertEqual(choose_human_action(ui, [(0, 5, 0), (0, 6, 0)]),
                         [0, 5, 0])

    def test_moving_and_capture(self):
        ui = FakeUI([click(4), click(5), click(12)])
        legal = [(4, 5, 12), (4, 5, 14), (3, 9, 0)]
        self.assertEqual(choose_human_action(ui, legal), [4, 5, 12])

    def test_invalid_click_ignored(self):
        ui = FakeUI([click(99), click(5)])
        self.assertEqual(choose_human_action(ui, [(0, 5, 0)]), [0, 5, 0])

    def test_quit(self):
        ui = FakeUI([{"type": "quit"}])
        with self.assertRaises(HumanQuit):
            choose_human_action(ui, [(0, 5, 0)])


if __name__ == "__main__":
    unittest.main()
