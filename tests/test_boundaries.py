"""Checks for temporal leakage, split isolation, and review-budget semantics."""
import unittest
import numpy as np
import pandas as pd
from prepare_data import eligible_cohort, aggregate_days, temporal_splits
from els import ranking_metrics, model_candidates, extend_features

class BoundaryTests(unittest.TestCase):
    def test_prediction_cutoff_and_unknown_outcomes(self):
        info=pd.DataFrame({'code_module':['BBB']*5,'code_presentation':['2013B']*5,
                           'id_student':[1,2,3,4,5],
                           'final_result':['Withdrawn','Withdrawn','Pass','Pass','Withdrawn']})
        reg=info[['code_module','code_presentation','id_student']].copy()
        reg['date_registration']=[-10,-10,1,-10,-10]
        reg['date_unregistration']=[27,28,None,None,None]
        result=eligible_cohort(info,reg)
        self.assertEqual(result.id_student.tolist(),[2,4])
        self.assertEqual(result.target.tolist(),[1,0])

    def test_future_activity_is_excluded(self):
        frame=pd.DataFrame({'code_presentation':['2013B']*4,'id_student':[1]*4,
                            'date':[-1,0,27,28],'sum_click':[900,2,3,9999]})
        result=aggregate_days(frame).iloc[0]
        self.assertEqual(result.total_clicks,5)
        self.assertEqual(result.active_days,2)
        self.assertEqual(result.days_since_activity,0)

    def test_review_budget_rounds_up(self):
        metrics=ranking_metrics([1,0,0,1,0,0,0,0,0,0,0],np.arange(11)[::-1])
        self.assertEqual(metrics['review_budget'],2)
        self.assertEqual(metrics['hits'],1)
        self.assertEqual(metrics['recall_at_10pct'],.5)

    def test_repeated_learners_are_removed_from_later_splits(self):
        frame=pd.DataFrame({'code_presentation':['2013B']*2+['2013J']*3+['2014J']*3,
                            'id_student':[1,2,1,3,4,3,5,6],
                            'target':[0,1,1,0,1,1,0,1]})
        parts,audit=temporal_splits(frame)
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            self.assertTrue(set(parts[a].id_student).isdisjoint(parts[b].id_student))
        self.assertEqual(audit.excluded_repeat_learners.tolist(),[0,1,1])

    def test_nonfinite_scores_are_rejected(self):
        with self.assertRaises(ValueError): ranking_metrics([0,1],[0,np.nan])

    def test_ties_have_a_deterministic_budget(self):
        result=ranking_metrics([1,0,1,0],np.ones(4),fraction=.5)
        self.assertEqual(result['review_budget'],2)
        self.assertEqual(result['hits'],1)

    def test_zero_activity_does_not_produce_infinite_intensity(self):
        frame=pd.DataFrame({'clicks_w1':[0],'clicks_w2':[0],'clicks_w3':[0],
                            'clicks_w4':[0],'total_clicks':[0],'active_days':[0]})
        self.assertTrue(pd.isna(extend_features(frame).clicks_per_active_day.iloc[0]))

    def test_benchmark_budget_and_outcome_free_features(self):
        candidates=model_candidates()
        self.assertEqual(len(candidates),21)
        for family in set(k[0] for k in candidates):
            self.assertEqual(sum(k[0]==family for k in candidates),3)
        from els import FEATURES
        self.assertTrue(set(FEATURES).isdisjoint({'target','id_student','date_unregistration','final_result'}))

    def test_inference_rejects_inconsistent_aggregates(self):
        from predict import validate_inputs
        row=pd.DataFrame({'clicks_w1':[1],'clicks_w2':[2],'clicks_w3':[3],
                          'clicks_w4':[4],'total_clicks':[99],'active_days':[4],
                          'days_since_activity':[0],'last_week_change':[1]})
        with self.assertRaises(ValueError):validate_inputs(row)

if __name__=='__main__': unittest.main()
