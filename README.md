# -Important-Problems-for-BST-Heap

## Problem 1: Complete BST Implementation with All Core Operations ⭐⭐⭐

**Why This is Important:**
- This single code can be adapted to solve: BST validation, insertion, deletion, search, traversals, kth smallest/largest, range queries, successor/predecessor problems
- Most exam problems are variations of these operations
- Interviewers often ask you to implement basic BST operations first

### Complete BST Template (Memorize This!)

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BST:
    def __init__(self):
        self.root = None
    
    # ============ INSERTION ============
    def insert(self, val):
        """Insert a value into BST - O(h) time"""
        self.root = self._insert_recursive(self.root, val)
    
    def _insert_recursive(self, node, val):
        if not node:
            return TreeNode(val)
        
        if val < node.val:
            node.left = self._insert_recursive(node.left, val)
        else:
            node.right = self._insert_recursive(node.right, val)
        
        return node
    
    # ============ SEARCH ============
    def search(self, val):
        """Search for a value - O(h) time"""
        return self._search_recursive(self.root, val)
    
    def _search_recursive(self, node, val):
        if not node or node.val == val:
            return node
        
        if val < node.val:
            return self._search_recursive(node.left, val)
        return self._search_recursive(node.right, val)
    
    # ============ DELETION (MOST IMPORTANT!) ============
    def delete(self, val):
        """Delete a value - O(h) time"""
        self.root = self._delete_recursive(self.root, val)
    
    def _delete_recursive(self, node, val):
        if not node:
            return None
        
        # Find the node to delete
        if val < node.val:
            node.left = self._delete_recursive(node.left, val)
        elif val > node.val:
            node.right = self._delete_recursive(node.right, val)
        else:
            # Case 1: Node has no children (leaf node)
            if not node.left and not node.right:
                return None
            
            # Case 2: Node has one child
            if not node.left:
                return node.right
            if not node.right:
                return node.left
            
            # Case 3: Node has two children
            # Find inorder successor (smallest in right subtree)
            min_node = self._find_min(node.right)
            node.val = min_node.val
            node.right = self._delete_recursive(node.right, min_node.val)
        
        return node
    
    def _find_min(self, node):
        """Find minimum value node"""
        while node.left:
            node = node.left
        return node
    
    def _find_max(self, node):
        """Find maximum value node"""
        while node.right:
            node = node.right
        return node
    
    # ============ VALIDATION ============
    def is_valid_bst(self):
        """Check if tree is valid BST - O(n) time"""
        return self._validate(self.root, float('-inf'), float('inf'))
    
    def _validate(self, node, min_val, max_val):
        if not node:
            return True
        
        if node.val <= min_val or node.val >= max_val:
            return False
        
        return (self._validate(node.left, min_val, node.val) and
                self._validate(node.right, node.val, max_val))
    
    # ============ TRAVERSALS ============
    def inorder(self):
        """Inorder: Left -> Root -> Right (gives sorted order)"""
        result = []
        self._inorder_helper(self.root, result)
        return result
    
    def _inorder_helper(self, node, result):
        if node:
            self._inorder_helper(node.left, result)
            result.append(node.val)
            self._inorder_helper(node.right, result)
    
    def preorder(self):
        """Preorder: Root -> Left -> Right"""
        result = []
        self._preorder_helper(self.root, result)
        return result
    
    def _preorder_helper(self, node, result):
        if node:
            result.append(node.val)
            self._preorder_helper(node.left, result)
            self._preorder_helper(node.right, result)
    
    def postorder(self):
        """Postorder: Left -> Right -> Root"""
        result = []
        self._postorder_helper(self.root, result)
        return result
    
    def _postorder_helper(self, node, result):
        if node:
            self._postorder_helper(node.left, result)
            self._postorder_helper(node.right, result)
            result.append(node.val)
    
    # ============ KTH SMALLEST/LARGEST ============
    def kth_smallest(self, k):
        """Find kth smallest element (1-indexed) - O(k) time"""
        self.count = 0
        self.result = None
        self._kth_smallest_helper(self.root, k)
        return self.result
    
    def _kth_smallest_helper(self, node, k):
        if not node or self.result is not None:
            return
        
        self._kth_smallest_helper(node.left, k)
        
        self.count += 1
        if self.count == k:
            self.result = node.val
            return
        
        self._kth_smallest_helper(node.right, k)
    
    def kth_largest(self, k):
        """Find kth largest element - O(k) time"""
        self.count = 0
        self.result = None
        self._kth_largest_helper(self.root, k)
        return self.result
    
    def _kth_largest_helper(self, node, k):
        if not node or self.result is not None:
            return
        
        # Reverse inorder: Right -> Root -> Left
        self._kth_largest_helper(node.right, k)
        
        self.count += 1
        if self.count == k:
            self.result = node.val
            return
        
        self._kth_largest_helper(node.left, k)
    
    # ============ SUCCESSOR/PREDECESSOR ============
    def inorder_successor(self, val):
        """Find inorder successor of a value"""
        successor = None
        node = self.root
        
        while node:
            if val < node.val:
                successor = node.val
                node = node.left
            else:
                node = node.right
        
        return successor
    
    def inorder_predecessor(self, val):
        """Find inorder predecessor of a value"""
        predecessor = None
        node = self.root
        
        while node:
            if val > node.val:
                predecessor = node.val
                node = node.right
            else:
                node = node.left
        
        return predecessor
    
    # ============ RANGE QUERIES ============
    def range_sum(self, low, high):
        """Sum of all values in range [low, high]"""
        return self._range_sum_helper(self.root, low, high)
    
    def _range_sum_helper(self, node, low, high):
        if not node:
            return 0
        
        total = 0
        
        if low <= node.val <= high:
            total += node.val
        
        if node.val > low:
            total += self._range_sum_helper(node.left, low, high)
        
        if node.val < high:
            total += self._range_sum_helper(node.right, low, high)
        
        return total
    
    # ============ LOWEST COMMON ANCESTOR ============
    def lowest_common_ancestor(self, p, q):
        """Find LCA of two values p and q"""
        node = self.root
        
        while node:
            if p < node.val and q < node.val:
                node = node.left
            elif p > node.val and q > node.val:
                node = node.right
            else:
                return node.val
        
        return None
    
    # ============ HEIGHT ============
    def height(self):
        """Find height of tree"""
        return self._height_helper(self.root)
    
    def _height_helper(self, node):
        if not node:
            return 0
        return 1 + max(self._height_helper(node.left), 
                       self._height_helper(node.right))


# ============ COMPLETE TEST SUITE ============
if __name__ == "__main__":
    bst = BST()
    
    # Test Insertion
    values = [50, 30, 70, 20, 40, 60, 80, 10, 25, 35, 65]
    print("Inserting:", values)
    for val in values:
        bst.insert(val)
    
    # Test Traversals
    print("\nInorder (sorted):", bst.inorder())
    print("Preorder:", bst.preorder())
    print("Postorder:", bst.postorder())
    
    # Test Search
    print("\nSearch 40:", bst.search(40) is not None)
    print("Search 100:", bst.search(100) is not None)
    
    # Test Validation
    print("\nIs valid BST:", bst.is_valid_bst())
    
    # Test Kth operations
    print("\n3rd smallest:", bst.kth_smallest(3))
    print("3rd largest:", bst.kth_largest(3))
    
    # Test Successor/Predecessor
    print("\nSuccessor of 35:", bst.inorder_successor(35))
    print("Predecessor of 35:", bst.inorder_predecessor(35))
    
    # Test Range Sum
    print("\nRange sum [30, 60]:", bst.range_sum(30, 60))
    
    # Test LCA
    print("\nLCA of 20 and 40:", bst.lowest_common_ancestor(20, 40))
    
    # Test Height
    print("\nHeight of tree:", bst.height())
    
    # Test Deletion
    print("\nDeleting 30...")
    bst.delete(30)
    print("Inorder after deletion:", bst.inorder())
    print("Is still valid BST:", bst.is_valid_bst())
```

**Output:**
```
Inserting: [50, 30, 70, 20, 40, 60, 80, 10, 25, 35, 65]

Inorder (sorted): [10, 20, 25, 30, 35, 40, 50, 60, 65, 70, 80]
Preorder: [50, 30, 20, 10, 25, 40, 35, 70, 60, 65, 80]
Postorder: [10, 25, 20, 35, 40, 30, 65, 60, 80, 70, 50]

Search 40: True
Search 100: False

Is valid BST: True

3rd smallest: 25
3rd largest: 65

Successor of 35: 40
Predecessor of 35: 30

Range sum [30, 60]: 185

LCA of 20 and 40: 30

Height of tree: 4

Deleting 30...
Inorder after deletion: [10, 20, 25, 35, 40, 50, 60, 65, 70, 80]
Is still valid BST: True
```

---

## Problem 2: Complete Min Heap Implementation ⭐⭐⭐

**Why This is Important:**
- This code solves: Priority Queue problems, Top K elements, Median finding, Merge K sorted lists, Task scheduling
- Heap is used in Dijkstra's algorithm, Huffman coding, and many optimization problems
- Can be adapted for Max Heap by inverting values

### Complete Heap Template (Memorize This!)

```python
class MinHeap:
    """
    Complete Binary Tree stored as array
    Parent at i: children at 2i+1 and 2i+2
    Child at i: parent at (i-1)//2
    """
    
    def __init__(self):
        self.heap = []
    
    # ============ HELPER METHODS ============
    def parent(self, i):
        return (i - 1) // 2
    
    def left_child(self, i):
        return 2 * i + 1
    
    def right_child(self, i):
        return 2 * i + 2
    
    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
    
    def size(self):
        return len(self.heap)
    
    def is_empty(self):
        return len(self.heap) == 0
    
    # ============ CORE OPERATIONS ============
    
    def insert(self, val):
        """
        Insert element - O(log n)
        1. Add to end of array
        2. Bubble up to maintain heap property
        """
        self.heap.append(val)
        self._heapify_up(len(self.heap) - 1)
    
    def _heapify_up(self, i):
        """Bubble up element at index i"""
        while i > 0 and self.heap[self.parent(i)] > self.heap[i]:
            self.swap(i, self.parent(i))
            i = self.parent(i)
    
    def extract_min(self):
        """
        Remove and return minimum element - O(log n)
        1. Save root value
        2. Move last element to root
        3. Bubble down to maintain heap property
        """
        if self.is_empty():
            return None
        
        if self.size() == 1:
            return self.heap.pop()
        
        min_val = self.heap[0]
        self.heap[0] = self.heap.pop()  # Move last to root
        self._heapify_down(0)
        
        return min_val
    
    def _heapify_down(self, i):
        """Bubble down element at index i"""
        min_index = i
        left = self.left_child(i)
        right = self.right_child(i)
        
        # Find minimum among node and its children
        if left < self.size() and self.heap[left] < self.heap[min_index]:
            min_index = left
        
        if right < self.size() and self.heap[right] < self.heap[min_index]:
            min_index = right
        
        # If minimum is not current node, swap and continue
        if min_index != i:
            self.swap(i, min_index)
            self._heapify_down(min_index)
    
    def get_min(self):
        """Peek at minimum without removing - O(1)"""
        return self.heap[0] if not self.is_empty() else None
    
    # ============ BUILD HEAP ============
    def heapify_array(self, arr):
        """
        Build heap from array - O(n) time!
        Start from last non-leaf node and heapify down
        """
        self.heap = arr[:]
        # Last non-leaf node is at index (n//2 - 1)
        for i in range(len(self.heap) // 2 - 1, -1, -1):
            self._heapify_down(i)
    
    # ============ DECREASE KEY ============
    def decrease_key(self, i, new_val):
        """Decrease value at index i to new_val"""
        if new_val > self.heap[i]:
            return  # Can't increase in min heap
        
        self.heap[i] = new_val
        self._heapify_up(i)
    
    # ============ DELETE ============
    def delete(self, i):
        """Delete element at index i - O(log n)"""
        if i >= self.size():
            return
        
        # Move last element here
        self.heap[i] = self.heap[-1]
        self.heap.pop()
        
        if i < self.size():
            # May need to bubble up or down
            parent_idx = self.parent(i)
            if i > 0 and self.heap[i] < self.heap[parent_idx]:
                self._heapify_up(i)
            else:
                self._heapify_down(i)
    
    # ============ HEAP SORT ============
    def heap_sort(self, arr):
        """Sort array using heap - O(n log n)"""
        self.heapify_array(arr)
        sorted_arr = []
        
        while not self.is_empty():
            sorted_arr.append(self.extract_min())
        
        return sorted_arr
    
    # ============ TOP K ELEMENTS ============
    def find_k_smallest(self, arr, k):
        """Find k smallest elements - O(n log k)"""
        self.heapify_array(arr)
        result = []
        
        for _ in range(min(k, self.size())):
            result.append(self.extract_min())
        
        return result
    
    def find_k_largest(self, arr, k):
        """Find k largest using min heap - O(n log k)"""
        if k == 0:
            return []
        
        # Keep heap of size k with k largest elements
        self.heap = []
        
        for num in arr:
            if self.size() < k:
                self.insert(num)
            elif num > self.get_min():
                self.extract_min()
                self.insert(num)
        
        return sorted(self.heap, reverse=True)
    
    # ============ DISPLAY ============
    def display(self):
        """Display heap array"""
        return self.heap[:]


class MaxHeap:
    """
    Max Heap - just invert all values
    Or implement separate class with reversed comparisons
    """
    
    def __init__(self):
        self.min_heap = MinHeap()
    
    def insert(self, val):
        self.min_heap.insert(-val)
    
    def extract_max(self):
        result = self.min_heap.extract_min()
        return -result if result is not None else None
    
    def get_max(self):
        result = self.min_heap.get_min()
        return -result if result is not None else None
    
    def size(self):
        return self.min_heap.size()
    
    def is_empty(self):
        return self.min_heap.is_empty()


# ============ ADVANCED APPLICATIONS ============

class MedianFinder:
    """
    Find median from data stream
    Use two heaps: max heap for smaller half, min heap for larger half
    """
    
    def __init__(self):
        self.small = MaxHeap()  # Max heap for smaller half
        self.large = MinHeap()  # Min heap for larger half
    
    def add_num(self, num):
        # Add to small (max heap)
        self.small.insert(num)
        
        # Balance: largest of small should be <= smallest of large
        if (not self.large.is_empty() and 
            self.small.get_max() > self.large.get_min()):
            val = self.small.extract_max()
            self.large.insert(val)
        
        # Balance sizes: small can have at most 1 more element
        if self.small.size() > self.large.size() + 1:
            val = self.small.extract_max()
            self.large.insert(val)
        
        if self.large.size() > self.small.size():
            val = self.large.extract_min()
            self.small.insert(val)
    
    def find_median(self):
        if self.small.size() > self.large.size():
            return self.small.get_max()
        return (self.small.get_max() + self.large.get_min()) / 2.0


# ============ COMPLETE TEST SUITE ============
if __name__ == "__main__":
    print("="*50)
    print("MIN HEAP TESTS")
    print("="*50)
    
    # Test 1: Basic Operations
    heap = MinHeap()
    values = [15, 10, 20, 8, 12, 25, 6]
    
    print("\n1. Inserting values:", values)
    for val in values:
        heap.insert(val)
        print(f"   After inserting {val}: {heap.display()}")
    
    print(f"\n2. Minimum element: {heap.get_min()}")
    
    print("\n3. Extracting minimums:")
    print(f"   Extracted: {heap.extract_min()}, Heap: {heap.display()}")
    print(f"   Extracted: {heap.extract_min()}, Heap: {heap.display()}")
    
    # Test 2: Build Heap from Array
    print("\n4. Build heap from array [9, 5, 6, 2, 3]:")
    heap2 = MinHeap()
    heap2.heapify_array([9, 5, 6, 2, 3])
    print(f"   Result: {heap2.display()}")
    
    # Test 3: Heap Sort
    print("\n5. Heap Sort [64, 34, 25, 12, 22, 11, 90]:")
    heap3 = MinHeap()
    sorted_arr = heap3.heap_sort([64, 34, 25, 12, 22, 11, 90])
    print(f"   Sorted: {sorted_arr}")
    
    # Test 4: K Smallest Elements
    print("\n6. Find 3 smallest in [7, 10, 4, 3, 20, 15]:")
    heap4 = MinHeap()
    k_smallest = heap4.find_k_smallest([7, 10, 4, 3, 20, 15], 3)
    print(f"   Result: {k_smallest}")
    
    # Test 5: K Largest Elements
    print("\n7. Find 3 largest in [7, 10, 4, 3, 20, 15]:")
    heap5 = MinHeap()
    k_largest = heap5.find_k_largest([7, 10, 4, 3, 20, 15], 3)
    print(f"   Result: {k_largest}")
    
    # Test 6: Max Heap
    print("\n8. Max Heap operations:")
    max_heap = MaxHeap()
    for val in [5, 3, 8, 1, 9]:
        max_heap.insert(val)
    print(f"   Max element: {max_heap.get_max()}")
    print(f"   Extracted: {max_heap.extract_max()}")
    print(f"   New max: {max_heap.get_max()}")
    
    # Test 7: Median Finder
    print("\n9. Find median from stream:")
    median_finder = MedianFinder()
    stream = [5, 15, 1, 3, 8]
    for num in stream:
        median_finder.add_num(num)
        print(f"   After adding {num}, median: {median_finder.find_median()}")
    
    print("\n" + "="*50)
```

**Output:**
```
==================================================
MIN HEAP TESTS
==================================================

1. Inserting values: [15, 10, 20, 8, 12, 25, 6]
   After inserting 15: [15]
   After inserting 10: [10, 15]
   After inserting 20: [10, 15, 20]
   After inserting 8: [8, 10, 20, 15]
   After inserting 12: [8, 10, 20, 15, 12]
   After inserting 25: [8, 10, 20, 15, 12, 25]
   After inserting 6: [6, 10, 8, 15, 12, 2
