from camri.data import BidsDataset, LegacyCamriLayout, Project


def test_bids_query_and_derivatives(tmp_path):
    root = tmp_path / "bids"
    (root / "sub-01" / "ses-01" / "func").mkdir(parents=True)
    (root / "derivatives" / "prep" / "sub-01" / "func").mkdir(parents=True)
    (root / "sub-01" / "ses-01" / "func" / "sub-01_ses-01_task-rest_bold.nii.gz").touch()
    (root / "derivatives" / "prep" / "sub-01" / "func" / "sub-01_desc-preproc_bold.nii.gz").touch()
    dataset = BidsDataset(root)
    result = dataset.query(subject="01", suffix="bold", extension=".nii.gz")
    assert len(result) == 1
    assert result[0].entities["task"] == "rest"
    project = Project.open(root)
    assert project.derivatives.pipelines == ["prep"]
    assert project.create_derivative("qc").is_dir()


def test_legacy_layout_query(tmp_path):
    (tmp_path / "proc" / "step").mkdir(parents=True)
    target = tmp_path / "proc" / "step" / "output.nii.gz"
    target.touch()
    assert LegacyCamriLayout(tmp_path).query(area="proc", extension=".nii.gz") == [target]
