import os

def merge_and_delete_files(directory):
    files = os.listdir(directory)
    files_to_merge = [f for f in files if " (1).csv" in f]

    for file_with_one in files_to_merge:
        base_file = file_with_one.replace(" (1).csv", ".csv")
        base_file_path = os.path.join(directory, base_file)
        file_with_one_path = os.path.join(directory, file_with_one)

        if os.path.exists(base_file_path):
            with open(base_file_path, 'r') as f_base:
                base_content = f_base.readlines()

            with open(file_with_one_path, 'r') as f_one:
                one_content = f_one.readlines()

            # Assuming the first line is the header
            if len(one_content) > 0 and len(base_content) > 0 and one_content[0] == base_content[0]:
                merged_content = base_content + one_content[1:]
            else:
                # If headers are different or missing, append all content
                merged_content = base_content + one_content

            with open(base_file_path, 'w') as f_base:
                f_base.writelines(merged_content)

            os.remove(file_with_one_path)
            print(f"Merged {file_with_one} into {base_file} and deleted {file_with_one}")
        else:
            print(f"Base file {base_file} not found for {file_with_one}. Skipping merge.")

# Specify the directory
directory_to_process = "/home/kpiotr6/Documents/stuida/praca_magisterska/proj/outputs_ready"
merge_and_delete_files(directory_to_process)
