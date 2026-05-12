from pathlib import Path

def get_rec_idfrom_filename(filename):
    return filename.split('.')[0]

def get_date_time_from_filename(filename):
    return filename.split('.')[1]

def get_year_from_rec_id(filename):
    date_time = get_date_time_from_filename(filename)
    return 2000 + int(date_time[:2])

def get_month_from_rec_id(filename):
    date_time = get_date_time_from_filename(filename)
    return int(date_time[2:4])

def get_day_from_rec_id(filename):
    date_time = get_date_time_from_filename(filename)
    return int(date_time[4:6])

def get_hour_from_rec_id(filename):
    date_time = get_date_time_from_filename(filename)
    return int(date_time[6:8])

def get_minute_from_rec_id(filename):
    date_time = get_date_time_from_filename(filename)
    return int(date_time[8:10])

def get_second_from_rec_id(filename):
    date_time = get_date_time_from_filename(filename)
    return int(date_time[10:12])

def get_all_rec_of_month(audio_dir: Path, year: int, month: int):
    recs = []
    audios = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.WAV"))
    for audio_file in audios:
        if get_year_from_rec_id(audio_file.name) == year and get_month_from_rec_id(audio_file.name) == month:
            recs.append(audio_file)
    return recs

if __name__ == "__main__":
    filename = "5756.220411103455.wav"
    print(filename)
    print(get_rec_idfrom_filename(filename))
    print(get_date_time_from_filename(filename))
    print(get_year_from_rec_id(filename))
    print(get_month_from_rec_id(filename))
    print(get_day_from_rec_id(filename))
    print(get_hour_from_rec_id(filename))
    print(get_minute_from_rec_id(filename))
    print(get_second_from_rec_id(filename))

    print("\nTesting get_all_rec_of_month:")
    audio_dir = Path("../soundbay/datasets/fannie_project/")
    year = 2022
    month = 4
    recs = get_all_rec_of_month(audio_dir, year, month)
    print(f"Found {len(recs)} recordings for {year}-{month:02d}")