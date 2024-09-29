import math

def find_optimal_grid(num_files):
    # 파일의 개수에 대한 제곱근 계산
    sqrt = math.sqrt(num_files)
    
    # 행과 열을 초기값으로 설정
    rows = math.floor(sqrt)
    cols = math.ceil(sqrt)
    
    # 행 * 열이 파일 수보다 작을 경우, 행 또는 열을 늘려줌
    while rows * cols < num_files:
        if cols > rows:
            rows += 1
        else:
            cols += 1
    
    return rows, cols

num_files = 31
rows, cols = find_optimal_grid(num_files)
print(f"행: {rows}, 열: {cols}")
