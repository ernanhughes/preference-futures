from preference_futures.editorial_mrq.specificity import render_specificity_audit


def test_render_specificity_audit_reports_dimensions_and_gates() -> None:
    arm = {
        "representation_size": 10,
        "maximum_l2_selections": 2,
        "mean_train_log_loss": 0.5,
        "mean_validation_log_loss": 0.6,
        "mean_test_log_loss": 0.7,
        "mean_train_test_gap": 0.2,
        "folds": [{}, {}],
    }
    report = {
        "arms": {
            "generic_unoriented": arm,
            "generic_choice_aware": arm,
            "mrq_blind": arm,
            "mrq_choice_aware": arm,
        },
        "dimension_comparisons": {
            "generic_choice_aware_to_mrq_choice_aware_ratio": 4.0,
            "generic_unoriented_to_mrq_blind_ratio": 3.0,
        },
        "interpretation_gate": {
            "dimension_matched_controls_required": True,
            "extended_l2_diagnostic_indicated": True,
            "note": "qualification",
        },
    }

    rendered = render_specificity_audit(report)

    assert "generic_choice_aware" in rendered
    assert "4.000" in rendered
    assert "Dimension-matched controls required: `True`" in rendered
    assert "Extended-L2 diagnostic indicated: `True`" in rendered
