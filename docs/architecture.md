# Architecture Documentation

## Pipeline Modules
- **Ingest**: Module responsible for acquiring data from various sources.
- **Pitch Mapping**: Converts data into a representation suitable for analysis on a football pitch.
- **Detection**: Identifies players, the ball, and other relevant objects from video streams.
- **Tracking**: Maintains the state and location of identified entities throughout the game.
- **World Coords**: Transforms 2D pitch coordinates to global reference coordinates.
- **Events**: Captures significant events during gameplay, such as goals, fouls, etc.
- **Overlays**: Visual representation of analysis on the video feed.
- **Storage**: Manages data persistence and retrieval.