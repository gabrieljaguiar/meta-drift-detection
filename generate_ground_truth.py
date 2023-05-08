from experimental_drifts import drifting_streams
from utils import concept_drift


stream_idx = 0
ground_truth_positions = []
for stream in drifting_streams:
    position = stream.position
    ground_truth_positions.append({"g": stream_idx, "idx": position})

    while type(stream.nextStream) is concept_drift.ConceptDriftStream:
        stream = stream.nextStream
        position += stream.position
        ground_truth_positions.append({"g": stream_idx, "idx": position})

    stream_idx += 0


print(ground_truth_positions)
