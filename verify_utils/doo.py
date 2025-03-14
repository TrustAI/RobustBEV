import numpy as np
from abc import ABC, abstractmethod


class Recorder:
    '''
        A recorder that stores all intermediate results.
    '''
    def __init__(self,
                 nb_var,
                 max_deep,
                 max_feval,
                 ):
        self.nb_var = nb_var
        # pre-comput trisection results to improve efficiency.
        self.thirds = tuple([(1./3)** i for i in range(1, max_deep + 1)])
        # initlize space for query results
        self.poses = [[] for i in range(max_feval)]
        self.lengths = np.zeros((max_feval, self.nb_var))
        self.levels = np.ones((max_feval, self.nb_var), dtype=int) * -1
        self.fc_vals = np.array([np.inf for i in range(max_feval)])
        self.sizes = np.zeros(max_feval)

        # initlize results
        self.minimum = np.inf
        self.best_idx = None

        # initilize the first centre point 
        self.last_center = -1
        self.po_idxs = [[0]]

    def new_center(self, pos, length, level, surnd_size):
        '''
            Record several properties for a new point.
        '''
        self.last_center += 1
        self.poses[self.last_center] = pos
        self.lengths[self.last_center] = length
        self.levels[self.last_center] = level
        self.sizes[self.last_center] = surnd_size
    
    def update_func_val(self, f_values):
        '''
            Record query results and update the minimum found so far.
        '''
        nb_fval = len(f_values)
        start_idx = self.last_center + 1 - nb_fval
        self.fc_vals[start_idx : self.last_center + 1] = f_values

        tmp_min_idx = np.argmin(f_values)
        if f_values[tmp_min_idx] < self.minimum:
            self.minimum = f_values[tmp_min_idx].item()
            self.best_idx = start_idx + tmp_min_idx

    def report(self, logger):
        logger.debug(f"###----------- Recorder Reports -----------###")
        nb_feval = self.last_center + 1
        logger.debug(f'[-] Current pos: {self.poses[:nb_feval]}')
        logger.debug(f'[-] Current query results: {[float(item) for item in self.fc_vals[:nb_feval]]}')
        logger.debug(f'[-] Current lengths: {[item.tolist() for item in self.lengths[:nb_feval]]}')
        logger.debug(f'[-] Current levels: {[item.tolist() for item in self.levels[:nb_feval]]}')
        logger.debug(f'[-] Current sizes: {[float(item) for item in self.sizes[:nb_feval]]}')
        logger.debug(f'[-] Potential Optimal Points: {self.po_idxs[-1]}')
        logger.debug('##########----------------------------##########\n')


class Recorder_MaxSlope(Recorder):
    '''
        A recorder that stores intermediate results plus the maximal slopes.
    '''
    def __init__(self,
                 nb_var,
                 max_deep,
                 max_feval,):
        super(Recorder_MaxSlope, self).__init__(
            nb_var,
            max_deep,
            max_feval)

        # lipschitz estimite
        self.local_Lip = np.zeros(max_feval) - 1

    def update_func_val_and_slope(self, f_values, parent_idx, dist):
        '''
            Record query results;
            update the minimum found so far;
            Compute and record the local slope; 
            Return the largest one to update the corresponding centres' slope.
        '''
        nb_fval = len(f_values)
        start_idx = self.last_center + 1 - nb_fval
        self.fc_vals[start_idx : self.last_center + 1] = f_values

        tmp_min_idx = np.argmin(f_values)
        if f_values[tmp_min_idx] < self.minimum:
            self.minimum = f_values[tmp_min_idx].item()
            self.best_idx = start_idx + tmp_min_idx

        parent_f_value = self.fc_vals[parent_idx]
        tmp_slope = np.abs(f_values - parent_f_value) / dist
        self.local_Lip[start_idx] = tmp_slope[0]
        self.local_Lip[start_idx+1] = tmp_slope[1]
        return tmp_slope.max()

    def report(self, logger):
        logger.debug(f"###----------- Recorder Reports -----------###")
        nb_feval = self.last_center + 1
        logger.debug(f'[-] Current pos: {self.poses[:nb_feval]}')
        logger.debug(f'[-] Current slope: {self.local_Lip[:nb_feval]}')
        logger.debug(f'[-] Current query results: {[float(item) for item in self.fc_vals[:nb_feval]]}')
        logger.debug(f'[-] Current lengths: {[item.tolist() for item in self.lengths[:nb_feval]]}')
        logger.debug(f'[-] Current levels: {[item.tolist() for item in self.levels[:nb_feval]]}')
        logger.debug(f'[-] Current sizes: {[float(item) for item in self.sizes[:nb_feval]]}')
        logger.debug(f'[-] Potential Optimal Points: {self.po_idxs[-1]}')
        logger.debug('##########----------------------------##########\n')

class DirectBase(ABC):
    '''
        A base class for DIRECT solver.
    '''
    def __init__(self, problem,
                       nb_var,
                       bounds,
                       max_iter,
                       max_deep,
                       max_feval,
                       tolerance,
                       quant=3,
                       debug = False,
                       metric = 'linf', # ['linf', 'l1']
                       **kwargs):

        self.problem = problem
        self.nb_var = nb_var
        self.max_iter = max_iter
        self.max_feval = max_feval + 1
        self.max_deep = max_deep
        self.tolerance = tolerance
        self.debug = debug
        self.quant = quant

        self.lip_factor = 1

        if metric == 'linf':
            self.delta = lambda lst : np.max(lst) # l_infinite norm
        elif metric == 'l1':
            self.delta = lambda lst : np.sum(lst) # l1 norm
        elif metric == 'l2':
            self.delta = lambda lst : np.linalg.norm(lst) # l2 normlse:
        else:
            raise NotImplementedError

        self.lb, self.ub = self._set_var_bound(bounds, self.nb_var)
        self.space_length = self.ub - self.lb

        for k, v in kwargs.items():
            setattr(self, k.lower(), v)
        
        self.rcd = Recorder_MaxSlope(self.nb_var, self.max_deep, self.max_feval)

    def solve(self):
        init_pos = [0.5]*self.nb_var
        init_length = [1.]*self.nb_var
        init_level = [0]*self.nb_var
        self.rcd.new_center(init_pos, init_length, init_level, self.delta(init_length))

        init_fval = self._query_func_val([init_pos])
        self.rcd.update_func_val(init_fval)

        self.nb_iter = 1
        self.runout_query = False
        self.reach_max_deep = False
        while self.nb_iter <= self.max_iter and not self.runout_query and not self.reach_max_deep:
            self._divide_space()
            self.local_low_bound = self.estimate_low_bound()
            cur_po_points = self._find_po()
            self.rcd.po_idxs.append(self._check_compute_resource(cur_po_points))

            if self.debug and self.nb_iter % 10 == 0:
                print(f'[-] {self.nb_iter} th iter: Global minimum: {self.rcd.minimum:.6f} (estimated lower bound: {self.local_low_bound:.6f}). Number of funcation evaluation: {self.rcd.last_center + 1}, found largest slope: {np.max(self.rcd.local_Lip)}')

            self.nb_iter += 1

    def get_opt_size(self):
        opt_size = np.sum(self.rcd.sizes[self.rcd.best_idx] * self.space_length)
        return opt_size
    
    def get_largest_slope(self):
        return np.max(self.rcd.local_Lip[:self.rcd.last_center])

    def estimate_low_bound(self):
        cur_largest_slope = self.get_largest_slope()
        local_size = self.get_opt_size()
        low_bound = self.rcd.minimum - self.lip_factor * cur_largest_slope * local_size
        return low_bound

    @abstractmethod
    def _divide_space(self):
        pass

    @abstractmethod
    def _find_po(self):
        pass

    def _check_compute_resource(self, po_points): 
        not_reach_max_deep = np.min(self.rcd.levels[po_points], axis=1) < self.max_deep
        if True in not_reach_max_deep:
            po_points = po_points[not_reach_max_deep]
            new_po_points = []
            available_nb_queries = self.max_feval - self.rcd.last_center - 1
            for pp in po_points:
                nb_divide_dim = np.sum(self.rcd.levels[pp] == np.min(self.rcd.levels[pp]))
                require_nb_queries = nb_divide_dim *2
                if available_nb_queries >= require_nb_queries:
                    new_po_points.append(pp)
                    available_nb_queries -= require_nb_queries
                else: 
                    self.runout_query = True
                    break
            return new_po_points
        else:
            self.reach_max_deep = True
            return None

    def _query_func_val(self, centers):
        """
        Sequence query.
            centers : A list of evaluate points.
        """
        points = self.to_actual_point(centers) 
        ans = self.problem(points)
        return ans

    @staticmethod
    def _set_var_bound(bounds, nb_var):
        bounds = np.array(bounds, dtype=float)
        if not (bounds.shape == (nb_var,2)):
            raise AssertionError(
                  f'The shape of bounds should be ({nb_var},2). But got {bounds.shape}')
        lb = bounds[:,0]
        ub = bounds[:,1]
        if True in (lb - ub > 0):
            raise AssertionError('Low bound is larger than upper bound.'+
                  ' The lower bound should be with index 0,' +
                  ' the upper bound should be with index 1.')
        return lb, ub

    def optimal_result(self):
        optimal_idx = self.rcd.best_idx
        optimal_ans = self.to_actual_point(self.rcd.poses[optimal_idx]) 
        return optimal_ans

    @staticmethod
    def _to_unit_square(point, lb, space_length):
        return (point - lb)/space_length

    def to_actual_point(self, pos):
        return self.space_length * pos + self.lb

class BaselineDIRECT(DirectBase):
    def _calc_lbound(self, h, sizes):
        h_size = sizes[h]
        lb = []
        for pp in range(len(h)):
            tmp_rects = h_size < self.rcd.sizes[h[pp]]
            if True in tmp_rects:
                tmp_f = self.rcd.fc_vals[h[tmp_rects]]
                tmp_size = self.rcd.sizes[h[tmp_rects]]
                tmp_lbs = (self.rcd.fc_vals[h[pp]] - tmp_f) / (self.rcd.sizes[h[pp]] - tmp_size)
                lb.append(np.max(tmp_lbs))
            else:
                lb.append(-np.inf)
        return np.array(lb)

    def _calc_ubound(self, h, sizes):
        h_size = sizes[h]
        ub = []
        for pp in range(len(h)):
            tmp_rects = h_size > self.rcd.sizes[[h[pp]]]
            if True in tmp_rects:
                tmp_f = self.rcd.fc_vals[h[tmp_rects]]
                tmp_size = self.rcd.sizes[h[tmp_rects]]
                tmp_ubs = (tmp_f - self.rcd.fc_vals[h[pp]]) / (tmp_size - self.rcd.sizes[h[pp]])
                ub.append(np.min(tmp_ubs))
            else:
                ub.append(np.inf)
        return np.array(ub)

    def _divide_space(self):
        dims_dict = {}
        all_new_points = []

        for cur_idx in self.rcd.po_idxs[-1]:
            new_slope = 0
            divide_dims = np.where(self.rcd.levels[cur_idx] == np.min(self.rcd.levels[cur_idx]))[0]
            tmp_third = self.rcd.thirds[self.rcd.levels[cur_idx, divide_dims[0]]]
            tmp_mark = len(all_new_points)
            for d in divide_dims:

                tmp_left = self.rcd.poses[cur_idx][:]
                tmp_left[d] -= tmp_third
                all_new_points.append(tmp_left)

                tmp_right = self.rcd.poses[cur_idx][:]
                tmp_right[d] += tmp_third
                all_new_points.append(tmp_right)

            dims_dict[cur_idx] = (divide_dims, tmp_mark, 2*len(divide_dims))

        all_f_values = self._query_func_val(all_new_points)

        for cur_idx in self.rcd.po_idxs[-1]:
            divide_dims, start_mark, num_results = dims_dict[cur_idx]
            f_values = all_f_values[start_mark:start_mark+num_results]
            new_points = all_new_points[start_mark:start_mark+num_results]
            tmp_third = self.rcd.thirds[self.rcd.levels[cur_idx, divide_dims[0]]]

            sorted_idx = np.argsort(f_values)
            divide_idx = sorted_idx//2
            divide_idx = sorted(np.unique(divide_idx), key=divide_idx.tolist().index)
            new_points = [[new_points[i*2],new_points[i*2+1]] for i in range(len(divide_dims))]

            for pair,pair_idx in enumerate(divide_idx):

                new_length = self.rcd.lengths[cur_idx].copy()
                new_length[divide_dims[divide_idx[:pair+1]]] /= 3.
                new_size = self.delta(new_length)
                new_level = self.rcd.levels[cur_idx].copy()
                new_level[divide_dims[divide_idx[:pair+1]]] += 1

                self.rcd.new_center(new_points[pair_idx][0],
                                    new_length,
                                    new_level,
                                    new_size)
                self.rcd.new_center(new_points[pair_idx][1],
                                    new_length,
                                    new_level,
                                    new_size)

                tmp_fvals = [f_values[pair_idx*2], f_values[pair_idx*2 + 1]] 
                tmp_dist = (self.space_length[divide_dims[pair]]) * tmp_third
                tmp_slope = self.rcd.update_func_val_and_slope(tmp_fvals, cur_idx, tmp_dist)
                if tmp_slope > new_slope: new_slope = tmp_slope

            self.rcd.lengths[cur_idx][divide_dims] /= 3.
            self.rcd.levels[cur_idx][divide_dims] += 1
            self.rcd.sizes[cur_idx] = self.delta(self.rcd.lengths[cur_idx])
            self.rcd.local_Lip[cur_idx] = new_slope

    def _find_po(self):
        cur_sizes = self.rcd.sizes[:self.rcd.last_center+1]
        fc_vals_sub = self.rcd.fc_vals[:self.rcd.last_center+1]
        unique_sizes = np.unique(cur_sizes)

        hull = []
        for sz in unique_sizes:
            sz_idx = np.where(cur_sizes == sz)[0]
            sz_min_fval = np.min(fc_vals_sub[sz_idx])
            po_points = np.where(fc_vals_sub <= sz_min_fval)[0]

            intersection_points = np.intersect1d(po_points, sz_idx)
            hull.append(intersection_points)

        hull = np.concatenate(hull)
        lbound = self._calc_lbound(hull, cur_sizes)
        ubound = self._calc_ubound(hull, cur_sizes)
        po_cond1 = lbound - ubound <= 0

        po_cond2 = (self.rcd.minimum - self.rcd.fc_vals[hull] +
                    self.rcd.sizes[hull] * ubound)  >= self.tolerance * np.abs(self.rcd.minimum)

        po_cond = po_cond1 * po_cond2 
        return hull[po_cond]


class LowBoundedDIRECT_potential(BaselineDIRECT):
    '''
        Modified the find_po: add local slope into consideration.
    '''
    def _find_po(self):
        cur_sizes = self.rcd.sizes[:self.rcd.last_center+1]
        fc_vals_sub = self.rcd.fc_vals[:self.rcd.last_center+1]

        unique_sizes = np.unique(cur_sizes)

        hull = []
        for sz in unique_sizes:
            sz_idx = np.where(cur_sizes == sz)[0]
            sz_min_fval = np.min(fc_vals_sub[sz_idx])
            po_points = np.where(fc_vals_sub <= sz_min_fval)[0]
            intersection_points = np.intersect1d(po_points, sz_idx)
            hull.append(intersection_points)

        hull = np.concatenate(hull)
        ubound = self._calc_ubound(hull, cur_sizes)

        po_cond2 = (self.rcd.minimum - self.rcd.fc_vals[hull] +
            self.rcd.sizes[hull] * ubound)  >= (self.tolerance * np.abs(self.rcd.minimum))
        hull = hull[po_cond2]

        if len(hull) > self.quant:
        # if (self.nb_iter > self.max_deep
        #     and len(hull) > quantile*self.max_deep):
            po_fval = fc_vals_sub[hull[:-1]]
            # po_size = cur_sizes[hull[:-1]]
            po_slope = self.rcd.local_Lip[hull[:-1]]
            po_size = np.max(self.rcd.lengths[hull[:-1]]*self.space_length, axis=1)
            po_index = np.argsort(po_fval-po_size*po_slope)
            po_hull = hull[po_index[:self.quant-1]]
            hull = np.append(po_hull, hull[-1])
        return hull


# if __name__ == "__main__":

#     def schwefel_batch(X):
#         """
#         Compute the Schwefel function values for a batch of input vectors.

#         Parameters:
#         - X : numpy array of shape (batch_size, n)
#             Batch of input vectors.

#         Returns:
#         - numpy array of shape (batch_size,)
#             Array of Schwefel function values.
#         """
#         X = np.array(X)
#         n = X.shape[1]
#         return 418.9829 * n - np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1)

#     def ackley_batch(X):
#         """
#         Compute the Ackley function values for a batch of input vectors.

#         Parameters:
#         - X : numpy array of shape (batch_size, n)
#             Batch of input vectors.

#         Returns:
#         - numpy array of shape (batch_size,)
#             Array of Ackley function values.
#         """
#         X = np.array(X)
#         n = X.shape[1]
#         sum1 = np.sum(X**2, axis=1)
#         sum2 = np.sum(np.cos(2 * np.pi * X), axis=1)

#         term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum1 / n))
#         term2 = -np.exp(sum2 / n)
        
#         return term1 + term2 + 20 + np.exp(1)

#     def demo_problem(x_ins):
#         return schwefel_batch(x_ins)
#         # return ackley_batch(x_ins)

#     # kwargs = {'alpha':1, 'beta':1}
#     nb_dim = 5
#     b = [[-500., 500] for _ in range(nb_dim)]
#     # b = [[-5., 10] for _ in range(nb_dim)]

#     solver = BaselineDIRECT(demo_problem, nb_dim, b, max_deep=7,
#                                          max_feval=int(5e4), tolerance=1e-3, quant=3,
#                                          max_iter=3000, debug=True, metric='l2')
#     solver.solve()
#     print(solver.rcd.best_idx)
#     print(solver.rcd.minimum)
#     print(solver.optimal_result())

#     solver = SOO(demo_problem, nb_dim, b, max_deep=7,
#                              max_feval=int(5e4), tolerance=1e-3,
#                              max_iter=3000, debug=True, metric='l2')
#     solver.solve()
#     print(solver.rcd.best_idx)
#     print(solver.rcd.minimum)
#     print(solver.optimal_result())


