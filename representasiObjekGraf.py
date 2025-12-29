import cv2
import numpy as np
import math

# ==========================================
# 1. IMPLEMENTASI ALGORITMA BRESENHAM
# ==========================================
def bresenham_line(img, x0, y0, x1, y1, color=(0, 255, 0)):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    while True:
        # Cek apakah koordinat berada dalam batas gambar agar tidak error
        if 0 <= x0 < img.shape[1] and 0 <= y0 < img.shape[0]:
            img[y0, x0] = color
            
        if x0 == x1 and y0 == y1:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

# ==========================================
# 2. FUNGSI MATRIKS TRANSFORMASI
# ==========================================
def get_translation_matrix(tx, ty, tz):
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])

def get_scale_matrix(sx, sy, sz):
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])

def get_rotation_matrix_y(angle):
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ])

def get_rotation_matrix_x(angle):
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ])

# ==========================================
# 3. LOAD DATA (Vertex & Edge)
# ==========================================
def load_data(vert_file, edge_file):
    vertices = []
    edges = []
    
    # Membaca Vertices
    with open(vert_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 3:
                # Menambahkan 1 di akhir untuk koordinat homogen (x, y, z, 1)
                v = [float(p) for p in parts] + [1.0] 
                vertices.append(v)
                
    # Membaca Edges
    with open(edge_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:
                edges.append((int(parts[0]), int(parts[1])))
                
    return np.array(vertices), edges

# ==========================================
# 4. MAIN PROGRAM
# ==========================================
def main():
    # Setup Canvas
    width, height = 800, 600
    
    # Load Vertex dan Edge dari file TXT
    # UPDATE: Nama file disesuaikan menjadi data_vertex.txt dan data_edge.txt
    try:
        vertices, edges = load_data('data_vertex.txt', 'data_edge.txt')
    except FileNotFoundError:
        print("Error: File data_vertex.txt atau data_edge.txt tidak ditemukan.")
        return

    angle = 0
    
    while True:
        # Buat canvas hitam kosong
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # --- PROSES TRANSFORMASI ---
        # 1. Rotasi (agar objek berputar)
        rot_y = get_rotation_matrix_y(angle)
        rot_x = get_rotation_matrix_x(angle * 0.5)
        
        # 2. Skala (Memperbesar objek asli yang koordinatnya cuma 1)
        scale = get_scale_matrix(100, 100, 100)
        
        # 3. Translasi (Memindahkan ke tengah layar)
        trans = get_translation_matrix(width // 2, height // 2, 0)
        
        # Gabungkan matriks: Translasi * Rotasi * Skala
        # Urutan perkalian matriks sangat penting: Scale -> Rotate -> Translate
        transform_matrix = trans @ rot_y @ rot_x @ scale 

        # Hitung posisi vertex baru
        projected_points = []
        for v in vertices:
            # Perkalian Matriks: (4x4) dot (4x1)
            transformed = transform_matrix @ v
            
            # Ambil x dan y saja untuk digambar di layar 2D
            x = int(transformed[0])
            y = int(transformed[1])
            projected_points.append((x, y))

        # --- VISUALISASI TEORI GRAF (G = (V, E)) ---
        
        # 1. Gambar Edges (Garis) menggunakan BRESENHAM
        # Representasi sisi graf yang menghubungkan node
        for edge in edges:
            idx1, idx2 = edge
            p1 = projected_points[idx1]
            p2 = projected_points[idx2]
            
            # Panggil fungsi Bresenham custom kita
            bresenham_line(img, p1[0], p1[1], p2[0], p2[1], color=(0, 255, 255)) # Warna Kuning

        # 2. Gambar Vertices (Titik/Node)
        # Representasi node dalam graf
        for p in projected_points:
            # Kita gunakan cv2.circle untuk mempertegas posisi vertex (titik merah)
            cv2.circle(img, p, 5, (0, 0, 255), -1)

        # Tampilkan Info
        cv2.putText(img, "3D Object Viewer - Bresenham & Graph Theory", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(img, f"Vertices (V): {len(vertices)} | Edges (E): {len(edges)}", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Tampilkan hasil
        cv2.imshow("3D Projection", img)

        # Update sudut rotasi untuk animasi
        angle += 1
        
        # Tekan 'ESC' untuk keluar
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()