# Real-Time Simulation Explanation

The project does not claim to have access to a live production e-commerce website. Instead, it uses a production-style real-time simulation.

## Correct wording

> This is not a live production e-commerce site. It is a production-style real-time simulation: I use real historical REES46 clickstream events, replay them in event-time order as micro-batches, write them to S3/Delta, and update trending/product analytics like a Kafka or Kinesis-based pipeline would.

## How to explain the layers

Historical batch path:

```text
REES46 CSV -> Bronze Parquet -> Iceberg batch analytics
```

Speed/simulation path:

```text
REES46 event replay -> Delta speed layer -> trending products/hourly volume
```

Serving path:

```text
User aggregates -> Hudi user profiles -> FastAPI/Streamlit
```

## Why this is valid

- The input events are real historical clickstream records.
- Each event has an `event_time` field.
- Events can be replayed in event-time order.
- Micro-batches simulate the arrival pattern of streaming systems.
- In production, the replay script would be replaced by Kafka, Kinesis, or website SDK events.

## Demo proof

Show this sequence:

1. Existing `live-events/` folder state.
2. Start the replay/dashboard action.
3. New event files appear in `live-events/`.
4. Delta trending or Streamlit metrics update.

Useful commands:

```bash
aws s3 ls s3://clickstream-analytics-akash/live-events/ --recursive | tail
aws s3 ls s3://clickstream-analytics-akash/speed/delta/trending/ --recursive | grep _delta_log | tail
```
