import numpy as np
from scipy.stats import f, t
from ..algebra import safe_divide
from ...utils.dmat import strip_termname, lrange
from ...utils import arr2mat
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
# from scipy.linalg import pinvh

class Anova:
    def __init__(self, model, typ="I"):
        """
        Initialize ANOVA with optional dataset for permutation.

        Parameters
        ----------
        model : OLS
            Fitted OLS model object.
        typ : {"I", "II", "III"} or int, default="I"
            Type of sum of squares.
        df : pandas.DataFrame, optional
            Original DataFrame used to fit the model (needed for permutation).
        y : ndarray, optional
            Response array used to fit the model (needed for permutation).
        """
        self.model = model
        self.df = model.df
        self.y = model.y_
        # Normalize typ
        if isinstance(typ, int): typ_int = typ
        else:
            s = str(typ).strip().upper()
            roman = {"I":1, "II":2, "III":3}
            typ_int = roman.get(s, int(s) if s.isdigit() else None)
            if typ_int not in (1,2,3):
                raise ValueError(f"Unknown typ: {typ}")
        self.typ = typ_int
        self.results_ = None

    def fit(self):
        """
        Compute ANOVA based on the selected type.
        """
        ss_terms, ss_resid, df_terms, df_resid = self._fit_common()
        
        if self.typ == 1:
            f_values = self._fit_type1(ss_terms, ss_resid, df_terms, df_resid)
        elif self.typ == 2:
            f_values = self._fit_type2(df_resid)
        elif self.typ == 3:
            f_values = self._fit_type3(df_resid)
        else:
            raise ValueError(f"Unknown ANOVA type: {self.typ}")
        terms = [strip_termname(t) for t in self.model.term_names[1:]]
        ss_factor, p_values = self._statistic_inference(f_values, df_terms, df_resid, ss_resid)
        self.results_ = {
            'ss_type': self.typ,
            'terms': tuple(terms),
            'ss_terms': ss_factor[1:],
            'df_terms': df_terms[1:],
            'ss_resid': ss_resid,
            'df_resid': df_resid,
            'f_values': f_values[1:],
            'p_values': p_values[1:],
        }
        
    def summary(self):
        """
        Return computed ANOVA table.

        Returns
        -------
        dict
            ANOVA table with Sum of Squares, Degrees of Freedom, F-value, and p-value.
        """
        if self.results_ is None:
            raise RuntimeError("Call fit() before summary()")
        return self.results_

    def _fit_common(self):
        df_resid = self.model.statistics_["dof"]["error"]
        ss_terms = self.model.statistics_["ss"]["column"]
        df_terms = np.array(self.model.statistics_["dof"]["column"])[:, None]
        ss_resid = self.model.statistics_["ss"]["error"][None, :]
        return ss_terms, ss_resid, df_terms, df_resid
    
    def _fit_partial(self, df_resid):
        X = self.model.X_
        dinfo = X.design_info
        coef = np.concatenate([self.model.intercept_.copy(), self.model.coef_.copy()], axis=0)
        inv_xtx = self.model.statistics_["inv_XtX"]
        resid = self.model.resid
        scale = np.sum(resid**2, axis=0) / df_resid
        cov = inv_xtx[np.newaxis, :, :] * scale[:, None, None]
        return (X, dinfo, coef, scale, cov)
    
    @staticmethod
    def _statistic_inference(f_values, df_terms, df_resid, ss_resid):
        """
        Common finalization step for all ANOVA types.
        """
        ss_factor = f_values * df_terms * ss_resid / df_resid
        p_values = (1 - f.cdf(f_values, df_terms, df_resid))
        return ss_factor, p_values

    def _fit_type1(self, ss_terms, ss_resid, df_terms, df_resid):
        """
        Compute Type I (Sequential) Sum of Squares.
        """
        ms_column = safe_divide(ss_terms, df_terms)
        ms_resid = safe_divide(ss_resid, df_resid)
        f_values = safe_divide(ms_column, ms_resid)
        return f_values
    
    def _fit_type2(self, df_resid):
        """
        Compute Type II (Sequential) Sum of Squares.
        """
        dmat, dinfo, coef, scale, cov = self._fit_partial(df_resid)
        
        f_values = []
        for term_1 in dinfo.terms:
            slc_1 = dinfo.slice(term_1)
            L1 = lrange(slc_1.start, slc_1.stop)
            L2 = []
            # current terms
            term_set = set(term_1.factors)
            # Second level loop cycle
            for term_2 in dinfo.terms:
                other_set = set(term_2.factors)
                if term_set.issubset(other_set) and not term_set == other_set:
                    # other_set is interaction
                    slc_2 = dinfo.slice(term_2)
                    L1.extend(lrange(slc_2.start, slc_2.stop))
                    L2.extend(lrange(slc_2.start, slc_2.stop))

            L1 = np.eye(dmat.shape[1])[L1]
            L2 = np.eye(dmat.shape[1])[L2]
            if L2.size:
                LVL = np.dot(np.dot(L1,cov[0]),L2.T)
                from scipy import linalg
                orth_compl,_ = linalg.qr(LVL)
                r = L1.shape[0] - L2.shape[0]
                # L1|2
                # use the non-unique orthogonal completion since L12 is rank r
                L12 = np.dot(orth_compl[:,-r:].T, L1)
            else:
                L12 = L1
                r = L1.shape[0]
            r_matrix = L12
            f_values.append(self._f_test(scale, coef, r_matrix))
        return np.stack(f_values, 0)

    def _fit_type3(self, df_resid):
        """
        Compute Type III (Partial) Sum of Squares.
        """
        dmat, dinfo, coef, scale, _ = self._fit_partial(df_resid)
        
        f_values = []
        for i, term in enumerate(dinfo.terms):
            cols = dinfo.slice(term)
            L1 = np.eye(dmat.shape[1])[cols]
            L12 = L1
            r = L1.shape[0]
            f_values.append(self._f_test(scale, coef, L12))
        return np.stack(f_values, 0)
        
    def _f_test(self, scale: np.ndarray, coef: np.ndarray, r_matrix: np.ndarray) -> np.ndarray:
        """
        Compute F-statistics for multiple targets in a fully vectorized manner.

        Parameters
        ----------
        scale : ndarray, shape (n_targets,)
            Residual variances (σ²) for each target/voxel.
        coef : ndarray, shape (n_features, n_targets)
            Estimated regression coefficients β for each target.
        r_matrix : ndarray, shape (n_contrasts, n_features)
            Contrast matrix R specifying J linear hypotheses.

        Returns
        -------
        F : ndarray, shape (n_targets,)
            F-statistics for each target, computed as:

            .. math::
            F_t = \frac{(R β)_t^T \bigl(R (X^T X)^{-1} R^T\bigr)^{-1} (R β)_t}{J},

            where J = number of contrasts (rows of R).
        """
        # Contrast estimates Rβ: shape (J, n_targets)
        Rb = r_matrix @ coef  # (J, n_targets)

        # Compute base covariance R (X^T X)^- R^T: shape (J, J)
        XtX_inv = self.model.statistics_["inv_XtX"]  # (p, p)
        cov_base = r_matrix @ XtX_inv @ r_matrix.T  # (J, J)

        # Broadcast scale to form per-target covariance: shape (n_targets, J, J)
        # Each cov[t] = scale[t] * cov_base
        cov = scale[:, None, None] * cov_base[None, :, :]

        # Invert each covariance matrix: shape (n_targets, J, J)
        invcov = np.linalg.pinv(cov)
        # invcov = np.stack([pinvh(c, atol=1e-15, rtol=1e-10) for c in cov], axis=0)

        # Compute numerator via Einstein summation:
        # num[t] = Rb[:, t]^T @ invcov[t] @ Rb[:, t]
        num = np.einsum('jt,tjk,jt->t', Rb, invcov, Rb)

        # Divide by number of contrasts J to get F-statistics
        F = num / r_matrix.shape[0]
        return F
    
    def t_test(self, contrast, two_sided=True):
        """
        Compute T-statistics and p-values for a linear contrast.

        Parameters
        ----------
        contrast : array_like, shape (n_features,)
            Contrast vector R.
        two_sided : bool, default=True
            If True, compute two-tailed p-values; otherwise one-tailed (upper).

        Returns
        -------
        t_vals : ndarray, shape (n_targets,)
            T-statistics.
        p_vals : ndarray, shape (n_targets,)
            P-values.
        """
        model = self.model
        X = model.X_
        resid = model.resid
        df_resid = model.statistics_["dof"]["error"]
        beta = np.vstack([model.intercept_, model.coef_])
        scale = np.sum(resid**2, axis=0) / df_resid  
        XtX_inv = self.model.statistics_["inv_XtX"]
        num = contrast @ beta  
        var = contrast @ XtX_inv @ contrast.T  
        se = np.sqrt(scale * var)  
        t_vals = num / se
        if two_sided:
            p_vals = 2 * (1 - t.cdf(np.abs(t_vals), df_resid))
        else:
            p_vals = 1 - t.cdf(t_vals, df_resid)

        return {"t_values":t_vals, "p_values":p_vals}
    
    def _permute_once(self, shuffle_col, rng):
        unique_subj = self.df[shuffle_col].unique()
        subject_indices_map = {subj: np.where(self.df[shuffle_col] == subj)[0] for subj in unique_subj}
        
        permuted_subj_indices = rng.permutation(len(unique_subj))
        yp = np.zeros_like(self.y)
        current_start_row = 0
        for i in range(len(unique_subj)):
            original_subject_index = permuted_subj_indices[i]
            subject_whose_data_to_take = unique_subj[original_subject_index]
            source_indices = subject_indices_map[subject_whose_data_to_take]
            num_rows_for_subject = len(source_indices)
            target_slice = slice(current_start_row, current_start_row + num_rows_for_subject)
            if self.y.ndim == 1:
                yp[target_slice] = self.y[source_indices]
            elif self.y.ndim == 2:
                yp[target_slice, :] = self.y[source_indices, :]
            else:
                raise NotImplementedError
            current_start_row += num_rows_for_subject
        return yp
    
    def permute(self, shuffle_col, contrast=None, n_perm=1000, seed=None, 
                two_sided=False, max_stat=False):
        """
        Permutation testing for ANOVA or a contrast.

        If contrast is None, permutes F-statistics; else T-statistics.
        shuffle_col : column name in self.df for subject labels.
        contrast : array_like or None
        """
        if self.df is None or self.y is None:
            raise ValueError("df and y must be provided for permutation tests")
        rng = np.random.default_rng(seed)

        if contrast is None:
            self.fit()
            obs = self.results_["f_values"]
        else:
            obs = self.t_test(contrast)["t_values"]

        n_terms, n_voxels = obs.shape
        null_dist = np.zeros((n_perm, n_terms, n_voxels))
        null_max = np.zeros((n_perm, n_terms))

        for p in range(n_perm):
            yp = self._permute_once(shuffle_col, rng)

            m = self.model.__class__(self.model.formula, self.df, yp).fit()
            a = Anova(m, typ=self.typ)
            a.fit()
            if contrast is None:
                stat = a.results_["f_values"]
            else:
                stat = a.t_test(contrast)["t_values"]
            if max_stat:
                null_max[p] = np.max(np.abs(stat) if two_sided else stat, axis=1)
            null_dist[p] = stat

        if max_stat:
            if two_sided:
                p_vals = np.mean(np.abs(null_max)[:, :, None] >= np.abs(obs)[None, :, :], axis=0)
            else:
                p_vals = np.mean(null_max[:, :, None] >= obs[None, :, :], axis=0)
            return null_max, obs, p_vals
        else:
            # Empirical p-values
            if two_sided:
                p_vals = np.mean(np.abs(null_dist) >= np.abs(obs)[None, :, :], axis=0)
            else:
                p_vals = np.mean(null_dist >= obs[None, :, :], axis=0)
            return null_dist, obs, p_vals
    
    @staticmethod
    def _arr2symmat(arr):
        mat = arr2mat(arr)
        mat += mat.T
        return mat
    
    @staticmethod
    def _arr2graph(obs, threshold):
        if len(obs.shape) == 1:
            obs = obs[None, :]
        mask_obs = np.zeros(obs.shape, dtype=bool)
        if isinstance(threshold, list):
            for i, t in enumerate(threshold):
                mask_obs[i][obs[i] >= t] = True
        else:
            mask_obs[obs >= threshold] = True
        mask_obs = mask_obs.astype(int)
        
        graph_obs = []
        mask_mats = []
        for mo in mask_obs:
            mo_mat = Anova._arr2symmat(mo)
            graph_obs.append(csr_matrix(mo_mat))
            mask_mats.append(mo_mat)
        return graph_obs, np.array(mask_mats)
    
    @staticmethod
    def _compute_cluster_size(graph_obs, mask_obs):
        """Computes size (number of edges) of connected components"""
        clusters = []
        edge_sizes = [] 
        for i, graph in enumerate(graph_obs):
            mask = mask_obs[i] 
            n_comp, labels = connected_components(csgraph=graph,
                                                  directed=False,
                                                  return_labels=True)
            term_clusters = []
            term_edge_sizes = []
            for comp_idx in range(n_comp):
                nodes_in_comp = np.where(labels == comp_idx)[0]
                if len(nodes_in_comp) > 1:
                    comp_mask = mask[np.ix_(nodes_in_comp, nodes_in_comp)]
                    num_edges = int(np.sum(np.triu(comp_mask, k=1)))
                    if num_edges > 0:
                       term_edge_sizes.append(num_edges)
                       edge_list = []
                       rows, cols = np.triu_indices_from(comp_mask, k=1)
                       for r, c in zip(rows, cols):
                           if comp_mask[r, c]:
                               original_r, original_c = nodes_in_comp[r], nodes_in_comp[c]
                               edge_list.append(tuple(sorted((original_r, original_c))))
                       term_clusters.append(np.array(edge_list))
            clusters.append(term_clusters)
            if term_edge_sizes:
                 edge_sizes.append(term_edge_sizes)
            else:
                 edge_sizes.append([0]) 
        return clusters, edge_sizes
    
    def permute_nbs(self, shuffle_col, contrast=None, alpha=0.05,
                    n_perm=1000, seed=None, two_sided=False):
        """
        Permutation testing for ANOVA or a contrast.

        If contrast is None, permutes F-statistics; else T-statistics.
        shuffle_col : column name in self.df for subject labels.
        contrast : array_like or None
        """
        if self.df is None or self.y is None:
            raise ValueError("df and y must be provided for permutation tests")
        rng = np.random.default_rng(seed)
        self.fit()
        df_resid = self.results_['df_resid']
        # Compute observed statistics
        if contrast is None:
            df_terms = self.results_['df_terms']
            obs = self.results_["f_values"]  # shape (n_terms, n_voxels)
            threshold = f.ppf(1 - alpha, dfn=df_terms, dfd=df_resid)
            
        else:
            tt = self.t_test(contrast, two_sided=two_sided)
            obs = tt["t_values"][None, :]
            if two_sided:
                threshold = t.ppf(1 - alpha / 2, df=df_resid)
            else:
                threshold = t.ppf(1 - alpha, df=df_resid)
            
        n_term, _ = obs.shape
        # observed suprathrehold mask and extract clusters of edges
        graph_obs, mask_obs = self._arr2graph(obs, threshold=threshold)
        obs_clusters_edges, obs_cluster_sizes = self._compute_cluster_size(graph_obs, mask_obs)
        
        null_max_size = np.zeros((n_perm, n_term), dtype=int)

        for p in range(n_perm):
            yp = self._permute_once(shuffle_col, rng)
            # rebuild design
            m_perm = self.model.__class__(self.model.formula, self.df, yp).fit()
            a_perm = Anova(m_perm, typ=self.typ)
            if contrast is None:
                a_perm.fit()
                stat_p = a_perm.results_["f_values"]
            else:
                stat_p = a_perm.t_test(contrast, two_sided=two_sided)["t_values"]
            graph_p, mask_p = self._arr2graph(stat_p, threshold=threshold)
            _, perm_cluster_sizes  = self._compute_cluster_size(graph_p, mask_p)
            for term_idx in range(n_term):
                if term_idx < len(perm_cluster_sizes) and perm_cluster_sizes[term_idx]:
                    null_max_size[p, term_idx] = np.max(perm_cluster_sizes[term_idx])
            
        cluster_p_values = []
        for term_idx in range(n_term):
            term_pvals = []
            current_obs_sizes = obs_cluster_sizes[term_idx] if term_idx < len(obs_cluster_sizes) else [0]
            for obs_size in current_obs_sizes:
                if obs_size == 0: # Skip if observed size is 0 (placeholder)
                    term_pvals.append(1.0) # P-value is 1 if observed size is 0
                    continue
                # Proportion of permutations where max size >= observed size
                # Add 1 to numerator and denominator for conservative p-value (prevents p=0)
                p_val = float((np.sum(null_max_size[:, term_idx] >= obs_size) + 1) / (n_perm + 1))
                term_pvals.append(p_val)
            cluster_p_values.append(term_pvals)
        return obs_clusters_edges, cluster_p_values, null_max_size, obs_cluster_sizes
