import os

def get_files_in_folder(folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")

    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"'{folder_path}' is not a directory.")

    file_paths = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            file_paths.append(file_path)

    return file_paths

def read_file_and_filter_values(files_in_folder, output_file):
    filtered_rows = []
    old_seen = set()

    iter = 1
    for input_file in files_in_folder:
        seen = set()
        indices = []
        print(f"Reading file {iter} of {len(files_in_folder)}")
        with open(input_file, 'r', encoding="latin-1") as file:
            lines = file.readlines()

        if (lines[0].find(',') == -1):
            split_char = ';'
        else:
            split_char = ','

        if (lines[0].find('cod_detector') != -1):
            id_column = 'cod_detector'
        elif (lines[0].find('id_detector') != -1):
            id_column = 'id_detector'
        else:
            print(f"id_detector or cod_detector not found in the header! {lines[0]}")
            return

        header = lines[0].strip().split(split_char)
        for target in ["latitud", "longitud", id_column]:
            if target in header:
                indices.append(header.index(target))
            else:
                if (target == id_column):
                    indices.append(0)
                else:
                    print(f"{target} not found in the header! {header}")
                    return

        for line in lines[1:]:
            values = line.strip().split(split_char)
            filtered_row = [values[i] for i in indices]
            new_line = ','.join(filtered_row)

            if new_line in seen: continue

            if new_line in old_seen:
                seen.add(new_line)
                if input_file == files_in_folder[-1]:
                    filtered_rows.append(new_line)
            elif input_file == files_in_folder[0]:
                seen.add(new_line)

        print(f"Finished reading file, seen {len(seen)}")
        old_seen = seen
        iter += 1

    with open(output_file, 'w') as file:
        file.write('\n'.join(filtered_rows))

def read_file_and_group_by_coordinates(files_in_folder, output_file):
    grouped_data = {}

    specified_coordinates = [
        # ("326", "-34.905897", "-56.136828"),
        # ("407", "-34.905423", "-56.137037"),
        # ("107", "-34.906172", "-56.136634"),
        # ("214", "-34.905612", "-56.135593"),
        # ("106", "-34.905325", "-56.134761"),
        # ("205", "-34.905258", "-56.134153"),
        # ("107", "-34.906759", "-56.140034"),
        ("107", "-34.911150", "-56.151809"),
        ("109", "-34.911150", "-56.151809"),
        ("205", "-34.911057", "-56.151196"),
        ("206", "-34.911057", "-56.151196"),
        ("305", "-34.911658", "-56.151278"),
        ("306", "-34.911658", "-56.151278"),
        ("404", "-34.913389", "-56.150044"),
        ("408", "-34.913389", "-56.150044"),
        ("205", "-34.914897", "-56.150400"),
        ("206", "-34.914897", "-56.150400"),
        ("307", "-34.915119", "-56.149044"),
        ("308", "-34.915119", "-56.149044"),
        ("309", "-34.915119", "-56.149044"),
        ("107", "-34.914572", "-56.153585"),
        ("106", "-34.915828", "-56.149900"),
        ("107", "-34.915828", "-56.149900"),
        ("103", "-34.916417", "-56.148942"),
        ("106", "-34.916417", "-56.148942"),
        ("107", "-34.916417", "-56.148942"),
    ]

    iter = 1

    for input_file in files_in_folder:
        print(f"Processing file {iter} of {len(files_in_folder)}")

        with open(input_file, 'r') as file:
            lines = file.readlines()

        if (lines[0].find(',') == -1):
            split_char = ';'
        else:
            split_char = ','

        if (lines[0].find('cod_detector') != -1):
            id_column = 'cod_detector'
        elif (lines[0].find('id_detector') != -1):
            id_column = 'id_detector'
        else:
            print(f"id_detector or cod_detector not found in the header! {lines[0]}")
            return

        indices = []
        header = lines[0].strip().split(split_char)
        required_fields = [id_column, "fecha", "hora", "latitud", "longitud", "volume"]
        for target in required_fields:
            if target in header:
                indices.append(header.index(target))
            else:
                if (target == id_column):
                    indices.append(0)
                else:
                    print(f"{target} not found in the header! {header}")
                    return

        for line in lines[1:]:
            values = line.strip().split(split_char)
            detector, fecha, hora, latitud, longitud, volume = (values[i] for i in indices)

            if (detector, latitud, longitud) in specified_coordinates:
                key = (fecha, hora)
                if key not in grouped_data:
                    grouped_data[key] = []
                grouped_data[key].append((detector, latitud, longitud, volume))

        iter += 1
        print(f"Finished processing file, total lines: {len(grouped_data)}")

    i = 0
    # for i in range(len(specified_coordinates)):
        # with open(f'{output_file}_{i}.csv', 'w') as file:
        #     print(f"Writing file {i} of {len(specified_coordinates)}")
        #     specified_coordinates_shuffled = specified_coordinates.copy()
        #     del specified_coordinates_shuffled[i]
        #     specified_coordinates_shuffled.append(specified_coordinates[i])
        #     for key, values in grouped_data.items():
        #         fecha, hora = key
        #         [an, me, di] = fecha.split('-')
        #         [ho, mi, se] = hora.split(':')
        #         sorted_values = sorted(values, key=lambda x: specified_coordinates_shuffled.index((x[0], x[1], x[2])))
        #         volumes = [value[3] for value in sorted_values]
        #         if (len(volumes) == len(specified_coordinates)):
        #             # file.write(f"{','.join(volumes)}\n")
        #             file.write(f"{an},{me},{di},{ho},{mi},{','.join(volumes)}\n")

    with open(f'{output_file}.csv', 'w') as file:
        print(f"Writing file of {len(specified_coordinates)}")
        specified_coordinates_shuffled = specified_coordinates.copy()
        for key, values in grouped_data.items():
            fecha, hora = key
            [an, me, di] = fecha.split('-')
            [ho, mi, se] = hora.split(':')
            sorted_values = sorted(values, key=lambda x: specified_coordinates_shuffled.index((x[0], x[1], x[2])))
            volumes = [value[3] for value in sorted_values]
            if (len(volumes) == len(specified_coordinates)):
                # file.write(f"{','.join(volumes)}\n")
                file.write(f"{an},{me},{di},{ho},{mi},{','.join(volumes)}\n")


if __name__ == "__main__":
    folder_path = "/home/franco/Documents/Facultad/HPC/old/csv"
    files_in_folder = get_files_in_folder(folder_path)
    print(files_in_folder)
    output_file_path = '/home/franco/Documents/Facultad/HPC/old/training_data/training_data'
    # read_file_and_filter_values(files_in_folder, 'coords.txt')
    # Paste output_file_path content in https://mobisoftinfotech.com/tools/plot-multiple-points-on-map/ (Toggle Show Point Numbers to know witch point is witch)
    read_file_and_group_by_coordinates(files_in_folder, output_file_path)

