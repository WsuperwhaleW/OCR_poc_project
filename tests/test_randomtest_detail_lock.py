"""The random test can pin the Detail, the way it pins the document and models."""
import unittest

import randomtest


def make_plan(**kwargs):
    args = dict(rounds=12, cases=['a', 'b', 'c'], readers=['r1', 'r2'],
                extractors=['', 'x'], text_models=['r1', 'x'],
                details=['low', 'medium', 'original'],
                modes=['single', 'agentic'], seed=11)
    args.update(kwargs)
    return randomtest.plan(**args)


class DetailLockTests(unittest.TestCase):
    def test_every_round_reads_at_the_locked_detail(self):
        for strategy in ('balanced', 'uniform', 'sequential'):
            for scope in ('full', 'ocr'):
                plan = make_plan(scope=scope, strategy=strategy,
                                 lock={'detail': 'original'})
                self.assertEqual({r['detail'] for r in plan['rounds']},
                                 {'original'}, (strategy, scope))
                self.assertEqual(plan['lock'], {'detail': 'original'})

    def test_uniform_pool_shrinks_to_one_detail(self):
        free = make_plan(scope='ocr', strategy='uniform')
        locked = make_plan(scope='ocr', strategy='uniform',
                           lock={'detail': 'low'})
        self.assertEqual(locked['scenarios'] * 3, free['scenarios'])

    def test_unknown_detail_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'not a Detail preset'):
            make_plan(lock={'detail': 'max'})

    def test_fields_scope_drops_the_lock(self):
        plan = make_plan(scope='fields', lock={'detail': 'low'})
        self.assertNotIn('detail', plan['lock'])
        self.assertEqual({r['detail'] for r in plan['rounds']}, {''})


if __name__ == '__main__':
    unittest.main()
