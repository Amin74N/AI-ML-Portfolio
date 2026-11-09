from pathlib import Path
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
# Paths:
PROJECT_PATH = Path(".")
DATASET_PATH = PROJECT_PATH / "data" / "MozillaCommonVoice_EnglishDataset"
AUDIO_FOLDER = DATASET_PATH / "audios"
TRANSCRIPT_FILE = DATASET_PATH / "ss-corpus-en.tsv"
# ----------------------------------------------------------------------------------------------------------------------
# Path Validation:
def check_paths():                                                     # Validating dataset paths
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset folder not found:\n{DATASET_PATH}")
    if not AUDIO_FOLDER.exists():
        raise FileNotFoundError(f"Audio folder not found:\n{AUDIO_FOLDER}")
    if not TRANSCRIPT_FILE.exists():
        raise FileNotFoundError(f"Transcript file not found:\n{TRANSCRIPT_FILE}")
    print("Dataset Loaded Successfully")
# ----------------------------------------------------------------------------------------------------------------------
# Load Transcript:
def load_transcript():                                                 # Loading transcript data
    transcript_df = pd.read_csv(TRANSCRIPT_FILE, sep="\t")
    transcript_df = transcript_df[["audio_id", "audio_file", "transcription"]]
    transcript_df = transcript_df.dropna(subset=["transcription"])
    transcript_df = transcript_df.reset_index(drop=True)
    return transcript_df
# ----------------------------------------------------------------------------------------------------------------------
# Build Dataset:
def extract_audio_info(row):                                           # Reading audio files & extracting audio metadata
    audio_path = AUDIO_FOLDER / row["audio_file"]
    audio, sample_rate = sf.read(audio_path)
    duration = len(audio) / sample_rate
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    return {"audio_id": row["audio_id"],
            "audio_file": row["audio_file"],
            "transcription": row["transcription"],
            "sample_rate": sample_rate,
            "num_samples": len(audio),
            "duration": duration,
            "channels": channels}
def build_dataset(transcript_df):                                      # Building a structured ASR dataset
    dataset = []
    for _, row in transcript_df.iterrows():
        audio_info = extract_audio_info(row)
        dataset.append(audio_info)
    processed_df = pd.DataFrame(dataset)
    processed_df.to_csv("processed_dataset_metadata.csv", index=False) # Exporting processed metadata to CSV
    return dataset
# ----------------------------------------------------------------------------------------------------------------------
# Sample Viewer:
def display_sample(sample):                                            # Displaying transcript samples
    print(f'Audio ID   : {sample["audio_id"]}')
    print(f'Audio File : {sample["audio_file"]}')
    print(f'Duration   : {sample["duration"]:.2f} Seconds')
    print(f'Sample Rate: {sample["sample_rate"]} Hz')
    print(f'Channels   : {sample["channels"]}')
    print(f'Transcript : {sample["transcription"]}')
# ----------------------------------------------------------------------------------------------------------------------
# Dataset Summary:
def generate_summary(dataset):                                         # Generating dataset summary
    total_files = len(dataset)
    total_duration = sum(item["duration"] for item in dataset)
    average_duration = total_duration / total_files
    total_transcripts = len([item["transcription"] for item in dataset if item["transcription"]])
    summary = {"total_files": total_files,
               "total_duration": total_duration,
               "average_duration": average_duration,
               "total_transcripts": total_transcripts}
    return summary
def display_summary(summary):
    print(f"Total Files       : {summary['total_files']}")
    print(f"Total Duration    : {summary['total_duration']:.2f} seconds")
    print(f"Average Duration  : {summary['average_duration']:.2f} seconds")
    print(f"Total Transcripts : {summary['total_transcripts']}")
# ----------------------------------------------------------------------------------------------------------------------
# Text Statistics:
def generate_text_statistics(dataset):                                 # Computing transcript statistics
    word_counts = []
    for item in dataset:
        word_count = len(str(item["transcription"]).split())
        word_counts.append(word_count)
    shortest_sentence = min(dataset, key=lambda x: len(x["transcription"].split()))
    longest_sentence = max(dataset, key=lambda x: len(x["transcription"].split()))
    average_words = sum(word_counts) / len(word_counts)
    statistics = {"shortest_sentence": shortest_sentence["transcription"],
                  "longest_sentence": longest_sentence["transcription"],
                  "average_words": average_words}
    return statistics
def display_statistics(statistics):
    print(f"Shortest Sentence: {statistics['shortest_sentence']}")
    print(f"Longest Sentence: {statistics['longest_sentence']}")
    print(f"Average Words : {statistics['average_words']:.2f}")
# ----------------------------------------------------------------------------------------------------------------------
# Plot Functions:
def plot_duration_distribution(dataset):                               # Visualizing audio duration distribution
    durations = []
    for item in dataset:
        durations.append(item["duration"])
    plt.figure(figsize=(8,5))
    plt.hist(durations, bins=50)
    plt.title("Audio Duration Distribution")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Number of Files")
    plt.savefig("Audio_Duration_Distribution.png", dpi=300, bbox_inches="tight")
    plt.show()
def plot_word_distribution(dataset):                                   # Visualizing transcript word count distribution
    word_counts = []
    for item in dataset:
        count = len(item["transcription"].split())
        word_counts.append(count)
    plt.figure(figsize=(8,5))
    plt.hist(word_counts, bins=50)
    plt.title("Transcript Word Count Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Number of Samples")
    plt.savefig("Transcript_Word_Count_Distribution.png", dpi=300, bbox_inches="tight")
    plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# Main:
check_paths()
transcript_df = load_transcript()
print(f"Number of rows: {len(transcript_df)}")
print(f"Number of columns: {len(transcript_df.columns)}")
print(f"Columns: {list(transcript_df.columns)}")
print("-------------------------------------------------------")
dataset = build_dataset(transcript_df)
display_sample(dataset[0])
print("-------------------------------------------------------")
summary = generate_summary(dataset)
display_summary(summary)
print("-------------------------------------------------------")
statistics = generate_text_statistics(dataset)
display_statistics(statistics)
plot_duration_distribution(dataset)
plot_word_distribution(dataset)
