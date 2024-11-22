import numpy as np
import globalmap as GL
import galois
import pickle
import random
from scipy.special import comb
from collections import Counter


GF2 = galois.GF(2)

class Code:
    def __init__(self,H_filename):
        self.load_code(H_filename)
        
    def gf2elim(self,M):
          m,n = M.shape
          i=0
          j=0
          record_col_exchange_index = []
          while i < m and j < n:
              #print(M)
              # find value and index of largest element in remainder of column j
              if np.max(M[i:, j]):
                  k = np.argmax(M[i:, j]) +i
            # swap rows
                  #M[[k, i]] = M[[i, k]] this doesn't work with numba
                  if k !=i:
                      temp = np.copy(M[k])
                      M[k] = M[i]
                      M[i] = temp              
              else:
                  if not np.max(M[i, j:]):
                      M = np.delete(M,i,axis=0) #delete a all-zero row which is redundant
                      m = m-1  #update according info
                      continue
                  else:
                      column_k = np.argmax(M[i, j:]) +j
                      temp = np.copy(M[:,column_k])
                      M[:,column_k] = M[:,j]
                      M[:,j] = temp
                      record_col_exchange_index.append((j,column_k))
          
              aijn = M[i, j:]
              col = np.copy(M[:, j]) #make a copy otherwise M will be directly affected
              col[i] = 0 #avoid xoring pivot row with itself
              flip = np.outer(col, aijn)
              M[:, j:] = M[:, j:] ^ flip
              i += 1
              j +=1
          return M,record_col_exchange_index
          
    def generator_matrix(self,parity_check_matrix):
          # H assumed to be full row rank to obtain its systematic form
          tmp_H = np.copy(parity_check_matrix)
          #reducing into row-echelon form and record column 
          #indices involved in swapping
          row_echelon_form,record_col_exchange_index = self.gf2elim(tmp_H)
          H_shape = row_echelon_form.shape
          # H is reduced into [I H_2]
          split_H = np.hsplit(row_echelon_form,(H_shape[0],H_shape[1])) 
          #Generator matrix in systematic form [H_2^T I] in GF(2)
          G1 = split_H[1].T
          G2 = np.identity(H_shape[1]-H_shape[0],dtype=int)
          G = np.concatenate((G1,G2),axis=1)
          #undo the swapping of columns in reversed order
          for i in reversed(range(len(record_col_exchange_index))):
              temp = np.copy(G[:,record_col_exchange_index[i][0]])
              G[:,record_col_exchange_index[i][0]] = \
                  G[:,record_col_exchange_index[i][1]]
              G[:,record_col_exchange_index[i][1]] = temp
          #verify ths syndrome equal to all-zero matrix
          Syndrome_result = parity_check_matrix.dot(G.T)%2
          if np.all(Syndrome_result==0):
            print("That's it, generator matrix created successfully with shape:",G.shape)
          else:
            print("Something wrong happened, generator matrix failed to be valid")     
          return G
    def load_code(self,H_filename):
    	# parity-check matrix; Tanner graph parameters
        with open(H_filename,'rt') as f:
            line= str(f.readline()).strip('\n').split(' ')
    		# get n and m (n-k) from first line
            n,m = [int(s) for s in line]
            #assigned manually for redundant check matrix otherwise
           
    #################################################################################################################
            var_degrees = np.zeros(n).astype(int) # degree of each variable node    
    		# initialize H
            H = np.zeros([m,n]).astype(int)
            line =  str(f.readline()).strip('\n').split(' ')
            line =  str(f.readline()).strip('\n').split(' ')
            line =  str(f.readline()).strip('\n').split(' ')
    
            var_edges = [[] for _ in range(0,n)]
            for i in range(0,n):
                line =  str(f.readline()).strip('\n').split(' ')
                var_edges[i] = [(int(s)-1) for s in line if s not in ['0','']]
                var_degrees[i] = len(var_edges[i])
                H[var_edges[i], i] = 1
               
        self.G = self.generator_matrix(H) 
        self.original_H = H
        #effective number of message bits in a codeword        
        self.k = self.G.shape[0] 
        self.n = self.G.shape[1]   
        # Heuristic optimization using simulated annealing
        # Simulated annealing parameters
        initial_temp = 1000
        cooling_rate = 0.995
        num_iterations = 1000
        # Adjust this factor to balance between minimizing 4-cycles and row weight variation
        beta  = 1     
        if GL.get_map('regular_matrix'):
            self.H = H
            self.check_matrix_col = H.shape[1]
        else:
            if GL.get_map('generate_extended_parity_check_matrix'):
                reduction_iteration = GL.get_map('reduction_iteration')
                counter_pre = Counter(np.sum(H,1))
                print(sorted(counter_pre.items()))
                #initial reduction of row weight
                unique_H = self.sparsify_parity_check_matrix(H.copy())
                for i in range(reduction_iteration):
                    #further reductin of row weigtht
                    unique_H = self.refined_sparsify_parity_check_matrix(unique_H.copy())           
                
                redundancy_factor = GL.get_map('redundancy_factor') #redundancy multiplier
                expected_dim = self.n-self.k                
                if unique_H.shape[0] > expected_dim:
                    min_H = self.sort_matrix_by_row_sum(unique_H)[:redundancy_factor*(self.n-self.k),:]
                else:
                    gap_rows = redundancy_factor*(self.n-self.k-unique_H.shape[0])
                    min_H = unique_H
                    if gap_rows > 0:
                        for i in range(gap_rows):
                            j = i%unique_H.shape[0]
                            min_H = np.vstack([min_H,unique_H[j:j+1]])
                H_optimized, _ = self.simulated_annealing_row_operations(min_H, initial_temp, cooling_rate, num_iterations, beta)
                H_optimized = np.unique(H_optimized,axis=0)
                _,expected_rank = self.row_reduce_gf(H_optimized)
                counter_pro = Counter(np.sum(H_optimized,1))
                print(f'Rank:{expected_rank},{sorted(counter_pro.items())}')
                self.print_matrix_info(H_optimized)
                n_dim = self.n               
                with open('../BCH_'+str(n_dim)+'_training/ckpts/extended_parity_check_matrix.pkl', 'wb') as file:
                    # Use pickle.dump() to write the object to the file
                    pickle.dump(H_optimized, file) 
            else:   
                n_dim = self.n 
                with open('../BCH_'+str(n_dim)+'_training/ckpts/extended_parity_check_matrix.pkl', 'rb') as file:
                    # Use pickle.dump() to write the object to the file
                    H_optimized = pickle.load(file) 
            self.H = H_optimized
            print(f"\n Matrix ({H.shape}) before optimization:")
            self.print_matrix_info(H)
            self.atrributes_assign(H)
            print('Summary of parity check matrix:')
            print('Row weight:',self.row_weight_list)
            print('col weight:',self.col_weight_list)
            print(f"\n Matrix ({self.H.shape}) after optimization:")
            self.print_matrix_info(H_optimized)
            self.atrributes_assign(H_optimized)
            print('Summary of parity check matrix:')
            print('Row weight:',self.row_weight_list)
            print('col weight:',self.col_weight_list)
            
    def sort_matrix_by_row_sum(self,matrix):
        # Calculate the row sums (number of 1's in each row)
        row_sums = np.sum(matrix, axis=1)     
        # Get the sorted indices based on row sums
        sorted_indices = np.argsort(row_sums)
        # Reorder the matrix rows based on the sorted indices
        sorted_matrix = matrix[sorted_indices] 
        return sorted_matrix
 
    def get_min_weight_submatrix(self, matrix):
        # Calculate the row weights (sum of ones in each row)
        matrix = np.unique(matrix,axis=0)
        row_weights = np.sum(matrix, axis=1)
        
        # Find the minimum row weight
        min_weight = np.min(row_weights)
        
        # Extract rows with the minimum row weight
        min_weight_rows = matrix[row_weights == min_weight]
        
        return min_weight_rows

    def print_matrix_info(self,matrix):
        num_cycles = self.count_4_cycles(matrix)
        row_mean, row_std = self.row_weight_variation(matrix)
        col_mean,col_std = self.column_weight_variation(matrix)
        _,rank = self.row_reduce_gf(matrix)
        print(f"Number of 4-cycles for min_H:{num_cycles} of Rank:{rank}")
        print(f"Row/col weight {row_mean:.1f}/{col_mean:.1f} std:{row_std:.1f}/{col_std:.1f}")
    
    def atrributes_assign(self,matrix):
        self.check_matrix_col = matrix.shape[1]
        self.check_matrix_row = matrix.shape[0]
        self.row_weight_list =np.sum(matrix,axis=1)
        self.col_weight_list =np.sum(matrix,axis=0)     
            
    def count_4_cycles(self,H):
        rows, cols = H.shape
        num_cycles = 0
    
        # Iterate over pairs of columns
        for i in range(cols):
            for j in range(i + 1, cols):
                common_non_zero_rows = np.where((H[:, i] == 1) & (H[:, j] == 1))[0]
                num_common = len(common_non_zero_rows)
                if num_common >= 2:
                    num_cycles += comb(num_common, 2, exact=True)                   
        return num_cycles

    def weight_variation(self,weights):
        mean_weight = np.mean(weights)
        std = np.std(weights)
        return mean_weight, std
    
    def row_weight_variation(self,H):
        row_weights = np.sum(H, axis=1)
        mean_weight, std = self.weight_variation(row_weights)
        return mean_weight, std
    
    def column_weight_variation(self,H):
        col_weights = np.sum(H, axis=0)
        mean_weight, std = self.weight_variation(col_weights)
        return  mean_weight, std
    
    def cost_function(self,H,beta):
        num_4_cycles = self.count_4_cycles(H)
        _,col_weight_var = self.column_weight_variation(H)       
        return num_4_cycles + beta * col_weight_var
    
    def get_row_cycle_contributions(self,H):
        rows, cols = H.shape
        row_contributions = np.zeros(rows)
        # Iterate over pairs of columns
        for i in range(cols):
            for j in range(i + 1, cols):
                common_non_zero_rows = np.where((H[:, i] == 1) & (H[:, j] == 1))[0]
                num_common = len(common_non_zero_rows)
                if num_common >= 2:
                    for row in common_non_zero_rows:
                        row_contributions[row] += comb(num_common - 1, 1, exact=True)  # Contribution to 4-cycles    
        return row_contributions        
        
    def simulated_annealing_row_operations(self,H, initial_temp, cooling_rate, num_iterations, beta):
        current_temp = initial_temp
        current_H = H.copy()
        current_cost = self.cost_function(current_H, beta)
        best_H = current_H.copy()
        best_cost = current_cost
    
        for i in range(num_iterations):
            print('.',end='')
            if (i+1)%100 == 0:
                print(f'\n{i+1}th\n')
            if current_temp <= 0:
                break
            # Create a new candidate solution
            new_H = current_H.copy()
            #row shifts for cyclic codes
            # Calculate row contributions to 4-cycles
            row_contributions = self.get_row_cycle_contributions(new_H)
            # Select rows based on their contributions
            if np.sum(row_contributions)==0:
                best_H = current_H
                best_cost = self.cost_function(current_H,beta)
                break
            row_probs = row_contributions / np.sum(row_contributions)
            row = np.random.choice(new_H.shape[0], p=row_probs)
            shift_amount = np.random.randint(1, new_H.shape[1])
            new_H[row] = np.roll(new_H[row], shift_amount)
            new_cost = self.cost_function(new_H,beta)        
            # Accept new solution if it's better or with a certain probability if it's worse
            if new_cost < best_cost or random.uniform(0, 1) < np.exp((current_cost - new_cost) / current_temp):
                current_H = new_H.copy()
                current_cost = new_cost          
            if current_cost < best_cost:
                print(current_cost)
                best_H = current_H.copy()
                best_cost = current_cost 
            # Cool down
            current_temp *= cooling_rate  
        return best_H, best_cost
 #***************************************************************************************************#       
    def filter_min_weight(self,matrix,expected_rank):
        unique_matrix = np.unique(matrix, axis=0)    
        while True:
            reduced_matrix = self.exclude_max_weight(unique_matrix)
            _,reduced_rank = self.row_reduce_gf(reduced_matrix)
            if reduced_rank == expected_rank:
                unique_matrix = reduced_matrix
            else:
                #adding rows to supplement deficit rank
                new_reduced_matrix = self.supplement_rows(unique_matrix,reduced_matrix,expected_rank,reduced_rank)
                break
        _,min_rank = self.row_reduce_gf(new_reduced_matrix)
        print(f'Min matrix: rank: {min_rank} shape: {new_reduced_matrix.shape}')
        return new_reduced_matrix

    def append_cyclic_shift_with_rank_check(self,matrix,expected_rank):
        _,previous_rank = self.row_reduce_gf(matrix)  # Calculate original matrix rank
        new_matrix = matrix.copy()   
        counter = previous_rank
        for i in range(3*expected_rank): # 3 times attempts in acquiring full rank parity check matrix
            # Randomly select a row index and cyclically shift it
            row_index = np.random.randint(0, matrix.shape[0])
            row = matrix[row_index]
            shift = np.random.randint(1, len(row))  # Random shift (excluding no shift)
            shifted_row = np.roll(row, shift)           
            # Temporarily append the shifted row
            temp_matrix = np.vstack([new_matrix, shifted_row])          
            # Check if the rank of the new matrix has increased
            _,new_rank = self.row_reduce_gf(temp_matrix)
            if new_rank > previous_rank:
                # Append the shifted row to the matrix if the rank increased
                new_matrix = temp_matrix
                counter += 1
                previous_rank = new_rank
            if counter == expected_rank:
                break
        return new_matrix
              
    def supplement_rows(self,original_matrix,reduced_matrix,expected_rank,reduced_rank):
        counter = Counter(np.sum(original_matrix,axis=1))
        # Get all the keys and sort them
        sorted_keys = sorted(counter.keys())
        max_current_key = max(np.sum(reduced_matrix,axis=1))
        for element in sorted_keys:
            if element > max_current_key:
                chosen_weight = element
                break
        filtered_matrix = self.collect_given_weight(original_matrix,chosen_weight)
        # Shuffle the rows
        np.random.shuffle(filtered_matrix)
        gap = 2*(reduced_matrix.shape[1] - reduced_rank)
        # Append the new row to the matrix
        replenished_matrix = np.vstack([reduced_matrix, filtered_matrix[:gap]])
        return replenished_matrix
            
        
    def collect_given_weight(self,matrix,weight):
        #matrix = self.remove_circular_duplicates(matrix)
        # Calculate the sum of each row
        new_matrix = matrix.copy()
        row_sums = np.sum(new_matrix, axis=1)  
        # Create a boolean mask where True corresponds to rows with the minimum sum
        mask = (row_sums == weight)
        # Use the mask to filter out the rows
        filtered_matrix = new_matrix[mask]
        return filtered_matrix        
        
    
    def exclude_max_weight(self,matrix):
        #matrix = self.remove_circular_duplicates(matrix)
        # Calculate the sum of each row
        new_matrix = matrix.copy()
        row_sums = np.sum(new_matrix, axis=1)
        # Find the minimum row sum
        max_row_sum = np.max(row_sums)   
        # Create a boolean mask where True corresponds to rows with the minimum sum
        mask = (row_sums != max_row_sum)
        # Use the mask to filter out the rows
        filtered_matrix = new_matrix[mask]
        return filtered_matrix
    
    def remove_rows_if_rank_unchanged(self,H_matrix):
        _,original_rank = self.row_reduce_gf(H_matrix)
        # Calculate row contributions to 4-cycles
        row_contributions = self.get_row_cycle_contributions(H_matrix)   
        # Get the indices that would sort the rows by descending row weight
        sorted_indices = np.argsort(-row_contributions)       
        # Rearrange the matrix based on these sorted indices
        sorted_matrix = H_matrix[sorted_indices]
        counter = 0
        temp_tensor = sorted_matrix.copy()
        while True:
            if counter >= temp_tensor.shape[0]:
                break
            temp_tensor = np.delete(H_matrix, counter, axis=0)        
            # Calculate the rank of the new tensor
            _,new_rank = self.row_reduce_gf(temp_tensor)            
            # If the rank is unchanged, remove the row; otherwise, continue
            if original_rank == new_rank:                   
                H_matrix = temp_tensor
            else:
                counter += 1
        return H_matrix
                
                
    def lexicographically_smallest_rotation(self,row):
        """Find the lexicographically smallest rotation of a row."""
        rotations = [np.roll(row, i) for i in range(len(row))]
        return min(rotations, key=lambda x: tuple(x))
    
    def remove_circular_duplicates(self,tensor):
        """Remove duplicate rows considering circular shifts."""
        normalized_rows = np.array([self.lexicographically_smallest_rotation(row) for row in tensor])
        unique_matrix = np.unique(normalized_rows, axis=0)
        counter_pro = Counter(np.sum(unique_matrix,1))
        print(sorted(counter_pro.items()))
        return unique_matrix

    def find_reduced_rows_initial(self,base_H,matrix_sum,range_start):
        row_sum_list = np.sum(matrix_sum,1)
        min_updated_value = min(row_sum_list[range_start+1:])
        new_row_list = []
        if np.sum(base_H[range_start]) <= min_updated_value:
            indices = [range_start]
            new_row_list.append(base_H[range_start])
        else:
            indices = [i for i, value in enumerate(row_sum_list) if value == min_updated_value]
            for j in indices:
                new_row_list.append(matrix_sum[j])
        return new_row_list

    def find_reduced_rows_advance(self,matrix_cmp,range_start):
        base_row = matrix_cmp[range_start:range_start+1]
        new_row_list = []
        minimal_value = np.sum(base_row)
        for shift in range(matrix_cmp.shape[1]):          
            matrix_sum = (base_row + np.roll(matrix_cmp,shift,axis=1))%2
            row_sum = np.sum(matrix_sum,1)
            # Check if the minimum value is zero
            # Find the minimum value of the list
            min_non_zero = np.min(row_sum[row_sum!=0])
            if min_non_zero <= np.sum(base_row):   
                minimal_value = min_non_zero
                # Get the indices of the minimum non-zero elements
                indices = np.where(row_sum == min_non_zero)[0]
                for j in indices:
                    element_tuple = (minimal_value,matrix_sum[j])
                    new_row_list.append(element_tuple)
        if minimal_value == np.sum(base_row):
            new_row_list.append((minimal_value,base_row.flatten()))
                    
        #check and keep the min elements only
        unique_filtered_list = []
        if new_row_list:
            # Find the minimum value with respect to the first element of all tuples
            min_value = min(new_row_list, key=lambda x: x[0])[0]          
            # Filter the list to include only tuples that have this minimum value
            filtered_list = [tup[1] for tup in new_row_list if tup[0] == min_value]
            #filtered_matrix = np.reshape(filtered_list,[-1,matrix_cmp.shape[1]])
            #print(np.sum(filtered_matrix,1))
            # Use a set to remove duplicates while preserving order
            seen = set()   
            for element in filtered_list:
                if tuple(element) not in seen:
                    unique_filtered_list.append(element)
                    seen.add(tuple(element))
        return unique_filtered_list

    def row_reduce_gf(self,matrix):
        GF_matrix = GF2(matrix)
        Ref_matrix = GF_matrix.row_reduce()
        Ref_rank = np.linalg.matrix_rank(Ref_matrix)
        return Ref_matrix,Ref_rank

    def sparsify_parity_check_matrix(self,H):
        Ref_matrix,Ref_rank = self.row_reduce_gf(H)
        #recover original reduced echelon form
        original_ref_H = Ref_matrix.view(np.ndarray)
        #print('Initial summary:')
        #print(f'Original rank:{Ref_rank}')
        new_row_list = []
        for i in range(original_ref_H.shape[0]):
            tmp_matrix = (Ref_matrix[i:i+1] + Ref_matrix).view(np.ndarray)
            # Find the minimum of non-zero elements
            row_sum = np.sum(tmp_matrix,1)
            min_non_zero = np.min(row_sum[row_sum!=0])
            if min_non_zero <= np.sum(original_ref_H[i]):
                # Get the indices of the minimum non-zero elements
                indices = np.where(row_sum == min_non_zero)[0]
                for j in indices:
                    new_row_list.append(tmp_matrix[j])
            if min_non_zero >= np.sum(original_ref_H[i]):
                new_row_list.append(original_ref_H[i])
        reduce_H_matrix = np.reshape(new_row_list,[-1,original_ref_H.shape[1]])
        # remove duplicated and shifted rows
        #unique_H_matrix = self.remove_rows_if_rank_unchanged(reduce_H_matrix)
        unique_H_matrix = self.remove_circular_duplicates(reduce_H_matrix)
        #verify ths syndrome equal to all-zero matrix
        syndrome_result = unique_H_matrix.dot(self.G.T)%2
        if np.all(syndrome_result==0):
          print("Initial parity checks passed !")
        else:
          print("Something wrong happened, parity check failed !")  
        return unique_H_matrix
    
    def refined_sparsify_parity_check_matrix(self,unique_ref_H):
        new_matrix_list = []
        for i in range(unique_ref_H.shape[0]):
            range_start = i
            row_candidates = self.find_reduced_rows_advance(unique_ref_H,range_start)
            new_matrix_list = new_matrix_list + row_candidates
        reduce_H_matrix = np.reshape(new_matrix_list,[-1,unique_ref_H.shape[1]])      
        # remove duplicated and shifted rows
        #unique_ref_H = self.remove_rows_if_rank_unchanged(reduce_H_matrix)
        unique_ref_H = self.remove_circular_duplicates(reduce_H_matrix)
        syndrome_result = unique_ref_H.dot(self.G.T)%2
        if np.all(syndrome_result==0):
          print("Advanced parity checks passed !")
        else:
          print("Something wrong happened, parity check failed !")  
        return unique_ref_H   
    
