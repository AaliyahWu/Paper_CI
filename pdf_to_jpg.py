"""
PDF 轉 JPG 批次轉檔工具
------------------------
功能：
1. 讀取指定資料夾內所有 PDF 檔案
2. 將每個 PDF 的每一頁轉成高畫質 JPG
3. 輸出到指定的輸出資料夾，並依「檔名_頁碼.jpg」命名

使用方式：
1. 安裝套件：pip install pymupdf
2. 修改下方 INPUT_FOLDER 與 OUTPUT_FOLDER 路徑
3. 執行：python pdf_to_jpg.py
"""

import fitz  # PyMuPDF
from pathlib import Path

# ========== 使用者設定區 ==========
INPUT_FOLDER = r"./pdfs"          # 放置 PDF 的資料夾路徑
OUTPUT_FOLDER = r"./jpg_output"   # 輸出 JPG 的資料夾路徑
DPI = 300                          # 畫質設定，數字越大越清晰、檔案也越大
                                    # 一般文件用 300 已經很清楚，若要印刷等級可用 600
JPG_QUALITY = 95                   # JPG 壓縮品質 (1-100)，95 已接近無損
# ===================================


def convert_pdf_to_jpg(pdf_path: Path, output_folder: Path, dpi: int, quality: int):
    """將單一 PDF 的每一頁轉成 JPG"""
    doc = fitz.open(pdf_path)
    zoom = dpi / 72  # PDF 預設是 72 DPI，換算成目標 DPI 的縮放倍率
    matrix = fitz.Matrix(zoom, zoom)

    pdf_name = pdf_path.stem  # 檔名(不含副檔名)

    for page_index in range(len(doc)):
        page = doc[page_index]
        pix = page.get_pixmap(matrix=matrix)  # 依縮放倍率渲染成點陣圖

        # 若 PDF 只有 1 頁，檔名就不加頁碼；多頁則加上頁碼避免覆蓋
        if len(doc) == 1:
            output_path = output_folder / f"{pdf_name}.jpg"
        else:
            output_path = output_folder / f"{pdf_name}_p{page_index + 1:03d}.jpg"

        pix.save(str(output_path), jpg_quality=quality)
        print(f"  已輸出：{output_path.name}")

    doc.close()


def main():
    input_folder = Path(INPUT_FOLDER)
    output_folder = Path(OUTPUT_FOLDER)
    output_folder.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_folder.glob("*.pdf"))

    if not pdf_files:
        print(f"在 {input_folder.resolve()} 找不到任何 PDF 檔案，請確認路徑是否正確。")
        return

    print(f"共找到 {len(pdf_files)} 個 PDF 檔案，開始轉檔（DPI={DPI}）...\n")

    for i, pdf_path in enumerate(pdf_files, start=1):
        print(f"[{i}/{len(pdf_files)}] 處理中：{pdf_path.name}")
        try:
            convert_pdf_to_jpg(pdf_path, output_folder, DPI, JPG_QUALITY)
        except Exception as e:
            print(f"  發生錯誤，略過此檔案：{e}")

    print("\n全部轉檔完成！")


if __name__ == "__main__":
    main()
