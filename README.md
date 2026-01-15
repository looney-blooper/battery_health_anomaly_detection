flowchart TB
    %% ===============================
    %% OFFLINE / MLOPS LAYER
    %% ===============================
    subgraph OFFLINE["Offline / OEM Environment (MLOps)"]
        DVC["Battery CSVs<br/>(DVC Tracked)"]
        PREP["Preprocessing<br/>(Frozen Contract)"]
        ZENML["ZenML Training Pipeline<br/>Dense AE<br/>LSTM (2-layer)<br/>Deep LSTM"]
        EVAL["Offline Evaluation (ZenML)<br/>Weekly Simulation<br/>Rolling Baseline<br/>🟢🟡🔴 Logic"]
        MLFLOW["MLflow Model Registry<br/>Versioning & Promotion"]

        DVC --> PREP
        PREP --> ZENML
        ZENML --> EVAL
        EVAL --> MLFLOW
    end

    %% ===============================
    %% EDGE LAYER
    %% ===============================
    subgraph EDGE["Edge / User Device"]
        SYNC["Model Sync<br/>(System Update)"]
        BASE["Base Model<br/>(Production)"]
        PERS["On-Device Personalization<br/>Encoder Only"]
        PMODEL["Personalized Model"]
        WEEKLY["Weekly Runtime Inference<br/>Rolling Baseline<br/>Notify 🔴 Only"]

        SYNC --> BASE
        BASE --> PERS
        PERS --> PMODEL
        PMODEL --> WEEKLY
    end

    %% ===============================
    %% CONNECTION
    %% ===============================
    MLFLOW --> SYNC


flowchart LR
    IN["Input Window<br/>(T × F)"]
    ENC["Encoder<br/>(Dense / LSTM)"]
    LAT["Latent Representation"]
    DEC["Decoder"]
    OUT["Reconstructed Window<br/>(T × F)"]
    ERR["Reconstruction Error<br/>(MSE)"]

    IN --> ENC --> LAT --> DEC --> OUT
    OUT --> ERR
