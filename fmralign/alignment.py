from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

from fmralign.pairwise import fit_pairwise
from fmralign.template import fit_template
from fmralign._utils import _check_labels


class Alignment(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        method="identity",
        target=None,
        labels=None,
        n_jobs=1,
        verbose=0,
    ):
        self.method = method
        self.target = target
        self.labels = labels
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y=None) -> None:
        self.labels_ = _check_labels(X[0], self.labels, verbose=self.verbose)

        if self.target is None:
            # Do template alignment
            self.fit_, self.template = fit_template(
                X,
                self.method,
                self.labels_,
                self.n_jobs,
                max(self.verbose - 1, 0),
            )
        elif isinstance(self.target, int):
            # Do pairwise alignment with a specific subject
            self.fit_ = fit_pairwise(
                X,
                self.target,
                self.method,
                self.labels_,
                self.n_jobs,
                max(self.verbose - 1, 0),
            )

    def _transform_one_array(self, X, method):
        # Check if method is fitted
        if not hasattr(method, "fit_"):
            raise ValueError(
                "This instance has not been fitted yet. "
                "Please call 'fit' before 'transform'."
            )
        return method.transform(X)

    def transform(self, X, subject_indices):
        return [
            self._transform_one_array(X[i], self.fit_[i])
            for i in subject_indices
        ]

    # Make inherited function harmless
    def fit_transform(self):
        """Parent method not applicable here.

        Will raise AttributeError if called.
        """
        raise AttributeError(
            "type object 'PairwiseAlignment' has no attribute 'fit_transform'"
        )

    def get_parcellation(self):
        """Get the parcellation masker used for alignment.

        Returns
        -------
        labels: `list` of `int`
            Labels of the parcellation masker.
        parcellation_img: Niimg-like object
            Parcellation image.
        """
        if hasattr(self, "parcel_masker"):
            check_is_fitted(self)
            labels = self.parcel_masker.get_labels()
            parcellation_img = self.parcel_masker.get_parcellation_img()
            return labels, parcellation_img
        else:
            raise AttributeError(
                (
                    "Parcellation has not been computed yet,"
                    "please fit the alignment estimator first."
                )
            )
