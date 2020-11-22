import math
arr1 = [-5,1,2,3,6]
expected_result = [0,-1,1,2,3,-5]
arr2 =[0,1,2,3]

def abs(a):
    return -a if a<0 else a
def re_sort(arr):
    if not arr:
        return
    arr_length = len(arr)
    if arr[0] > 0:
        return arr
    if arr[arr_length-1]<0:
        return reverse(arr)
    result = [0] * arr_length
    pivot = re_sort_helper(arr)
    position = 0
    result[position] = arr[pivot]
    position += 1
    start = pivot-1
    end = pivot +1
    while start>=0 or end < arr_length:
        if start < 0:
            temp_start = arr[0]
        else:
            temp_start = arr[start]
        if end >= arr_length:
            temp_end = arr[arr_length-1]
        else:
            temp_end = arr[end]
        if abs(temp_start) <= abs(temp_end):
            result[position] = temp_start
            start -=1
        else:
            result[position] = temp_end
            end += 1
        position += 1
    return result

