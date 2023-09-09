import numpy as np


def process_line(line: str):
    values = line.split(",")
    if not values[0]:
        values = values[1:]

    return list(map(float, values))


def get_time_str(index: int):
    return "%02d" % (index // 6) + "%02d" % (index % 6 * 10)


def run(all_files=False, override=True):
    generator = get_file_data("wind_df_08_05.csv")

    index = count = 0
    time2 = current = prev = None

    try:
        while True:
            index += 1

            if prev is None:
                time1, lats, lons, prev = next(generator)
                count += 1
            else:
                time1, prev = time2, current

            if time1.hour == 23:
                pass

            time2, lats, lons, current = next(generator)

            try:
                with open(f"wind_interp/interp_{time1.strftime('%H%M')}.csv", "w" if override else "x") as f:
                    save(prev, lons, lats, f)

            except FileExistsError:
                ...

            interp_data = interp(current, prev)
            count += 1

            if debug:
                print(interp_data, "\n")
                print(lats, sep="\n")
                print(lons)

            for i in range(5):
                time_str = get_time_str(index)
                index += 1

                try:
                    with open(f"wind_interp/interp_{time_str}.csv", "w" if override else "x") as f:
                        save(interp_data[i], lons, lats, f)

                except FileExistsError:
                    print(f"Skipping {time_str[:2]}:{time_str[2:]} interpolation")

            try:
                with open(f"wind_interp/interp_{time2.strftime('%H%M')}.csv", "w" if override else "x") as f:
                    save(current, lons, lats, f)

            except FileExistsError:
                ...

            if not all_files:
                break

    except StopIteration:
        print(index, count, "(Something's wrong...)" if index <= count else "(Success, hopefully)")


def save(data, first, second, file):
    print(end=",", file=file)
    for i, num in enumerate(second):
        print(num, end="," * (i < len(second) - 1), file=file)
    print(file=file)

    for i, row in enumerate(data):
        print(first[i], end=",", file=file)
        print(*row, sep=",", file=file)


def interp(current, prev):
    output = []

    for i in range(1, 6):
        copy = np.array(current)

        for x, row in enumerate(current):
            for y, val in enumerate(row):
                if x >= 3:
                    pass

                prev_val = prev[x][y]

                multiply = i / 6
                copy[x][y] = prev_val + (val - prev_val) * multiply

        output.append(copy)

    return output


debug = True

if __name__ == "__main__":
    from variables import get_file_data

    run(True)
