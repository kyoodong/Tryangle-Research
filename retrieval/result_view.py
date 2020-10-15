from PIL import Image

def main():
    save_dir = "output"

    with open('retrieval_result.txt', 'r') as f:
        fnames = f.readlines()

    index_type = fnames[0][:-1]

    Image.open(fnames[1][:-1]).save(f"{save_dir}/query_{index_type}.jpg")

    for i, fname in enumerate(fnames[2:], 1):
        path = fname[:-1]
        print(f"[INFO] {path} read...")
        Image.open(path).save(f"{save_dir}/result{i}_{index_type}.jpg")


if __name__ == '__main__':
    main()
