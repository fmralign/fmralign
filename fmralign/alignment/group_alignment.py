import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from fmralign.alignment.utils import (
    _check_input_arrays,
    _check_labels,
    _check_method,
    _check_target,
    _fit_template,
    _map_to_target,
)
from fmralign.methods import DetSRM


class GroupAlignment(BaseEstimator, TransformerMixin):
    """Performs group-level alignment of various subject data.

    This class aligns multiple subjects' data either to a computed template
    or to a specific target subject's data. It supports various alignment
    methods and can process data in parallel.

    Parameters
    ----------
    method : str or a `BaseAlignment` instance, default="identity"
        The alignment method to use. It can be a string representing the method name
        or an instance of a class derived from `BaseAlignment`. Available methods include:
        ["identity", "procrustes", "ot", "sparseuot", "ridge"].
    labels : array-like or None, default=None
        Describes each voxel label's in the case of non-overlapping parcels.
        If provided, local alignments can be performed in parallel.
        If None, global alignment is performed across all features.
    n_jobs : int, default=1
        Number of parallel jobs to run. -1 means using all processors.
    verbose : int, default=0
        Verbosity level. Higher values provide more detailed output.
    n_iter : int, default=2
        Number of iterations for the template alignment algorithm.
    scale_template : bool, default=False
        Whether to scale the features during template learning.
        If True, features are rescaled to the updating Euclidean mean.

    Attributes
    ----------
    labels_ : array-like
        Validated labels used during fitting.
    method_ : str
        Validated alignment method used during fitting.
    fit_ : list
        List of fitted alignment estimators, one per subject.
    template : array-like or None
        Computed template for template alignment. None for pairwise alignment.

    Examples
    --------
    >>> import numpy as np
    >>> from fmralign import GroupAlignment
    >>> n_voxels = 5

    >>> # Template alignment
    >>> alignment_dict = {
    ...     "sub-01": np.random.rand(10, n_voxels),
    ...     "sub-02": np.random.rand(10, n_voxels),
    ... }
    >>> testing_dict = {
    ...     "sub-01": np.random.rand(8, n_voxels),
    ...     "sub-02": np.random.rand(8, n_voxels),
    ... }
    >>> aligner = GroupAlignment(method="procrustes", n_iter=3)
    >>> aligner.fit(alignment_dict, y="template")
    >>> aligned_data = aligner.transform(testing_dict)

    >>> # Pairwise alignment to target
    >>> target_data = np.random.rand(10, n_voxels)
    >>> aligner = GroupAlignment(method="procrustes")
    >>> aligner.fit(alignment_dict, y=target_data)
    >>> aligned_data = aligner.transform(testing_dict)
    """

    def __init__(
        self,
        method="identity",
        labels=None,
        n_jobs=1,
        verbose=0,
        n_iter=2,
        scale_template=False,
    ):
        self.method = method
        self.labels = labels
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_iter = n_iter
        self.scale_template = scale_template
        self.template = None

    def fit(self, X, y="template"):
        """Fit the group alignment model to the data.

        Parameters
        ----------
        X : dict of array-like
            Dictionary where keys are subject identifiers and values are
            arrays of subject data. Each array should have the same number of
            samples and features.
        y : str or array-like, default="template"
            Alignment target, which determines the alignment strategy:

            - "template": Compute a single template from all subjects and align
              each subject to this shared template.
            - "leave_one_subject_out": Perform Leave-One-Subject-Out alignment. For each subject,
              compute a template from all other subjects, then align that subject
              to its corresponding template. This process repeats for every subject.
            - array-like: Align all subjects to the provided target data array.
        """
        # Validate input data
        self.subject_keys_, X_ = _check_input_arrays(X)
        alignment_strategy = _check_target(X_[0], y)
        self.labels_ = _check_labels(X_[0], self.labels)
        self.method_ = _check_method(self.method)

        if (
            isinstance(alignment_strategy, str)
            and alignment_strategy == "template"
        ):
            fit_, self.template = _fit_template(
                X_,
                self.method_,
                self.labels_,
                self.n_jobs,
                self.verbose,
                self.n_iter,
                self.scale_template,
            )
        elif (
            isinstance(alignment_strategy, str)
            and alignment_strategy == "leave_one_subject_out"
        ):
            fit_ = []
            for left_out_sub in self.subject_keys_:
                _, external_template = _fit_template(
                    [v for k, v in X.items() if k != left_out_sub],
                    self.method_,
                    self.labels_,
                    self.n_jobs,
                    self.verbose,
                    self.n_iter,
                    self.scale_template,
                )
                pairwise_fit_ = _map_to_target(
                    [X[left_out_sub]],
                    external_template,
                    self.method_,
                    self.labels_,
                    self.n_jobs,
                    self.verbose,
                )[0]
                fit_.append(pairwise_fit_)

        else:  # Pairwise alignment
            fit_ = _map_to_target(
                X_,
                np.asarray(y),
                self.method_,
                self.labels_,
                self.n_jobs,
                self.verbose,
            )

        self.fitted_estimators = dict(zip(self.subject_keys_, fit_))
        return self

    def _transform_one_array(self, X, estimator):
        """Transform a single subject's data using a fitted estimator.

        Parameters
        ----------
        X : array-like
            Subject data to transform. Should have the same number of
            voxels/columns as the data used during fitting.
        estimator : A fitted alignment estimator
            The fitted alignment estimator to use for transformation.

        Returns
        -------
        array-like
            Aligned subject data.

        Raises
        ------
        ValueError
            If the estimator has not been fitted yet.
        """
        return estimator.transform(X)

    def transform(self, X):
        """Transform the input arrays using the fitted model.

        Parameters
        ----------
        X : dict of array-like
            Dictionary where keys are subject identifiers and values are
            array of subject data. Each array should have the same number of
            samples and features.

        Returns
        -------
        dict of array-like
            Dictionary with transformed subject data.
        """
        if not hasattr(self, "fitted_estimators"):
            raise ValueError(
                "This instance has not been fitted yet. "
                "Please call 'fit' before 'transform'."
            )

        # Check if the keys are valid
        keys = list(X.keys())
        if not all(key in self.subject_keys_ for key in keys):
            raise ValueError(
                "Some subjects identifier are not present in the fitted model. "
                "Please check the input keys."
            )

        return {
            key: self._transform_one_array(X[key], self.fitted_estimators[key])
            for key in keys
        }

    # Make inherited function harmless
    def fit_transform(self):
        """Parent method not applicable here.

        Will raise AttributeError if called.
        """
        raise AttributeError(
            "type object 'GroupAlignment' has no 'fit_transform' attribute"
        )

    def predict_subject(self, X_test, Y_new):
        """Fit a new subject to the template and predict test data.

        Parameters
        ----------
        X_test : dict of array-like
            Dictionary where keys are subject identifiers and values are
            array of subject held-out data used to make a prediction on
            the template.
        Y_new : array-like
            Data used to fit the new subject to a pre-existing template.

        Returns
        -------
        array-like
            Predicted response on the new subject using the response
            computed on the template.
        """
        # Predict test data from Sources -> Template
        template_test = np.mean(list(self.transform(X_test).values()), axis=0)
        # Learn Template -> Target
        [pairwise_estimator] = _map_to_target(
            [self.template],
            Y_new,
            self.method_,
            self.labels_,
            self.n_jobs,
            self.verbose,
        )
        # Predict test data from Template -> Target
        if len(np.unique(self.labels_)) > 1 and isinstance(
            self.method_, DetSRM
        ):
            # Special case for Piecewise SRM
            # Data should be sent back to brain space
            y_pred = pairwise_estimator.transform(
                template_test, srm_space=False
            )
        else:
            y_pred = pairwise_estimator.transform(template_test)
        return y_pred
