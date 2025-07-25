# graphframes-rs

A Rust implementation of Apache Spark's GraphFrames using Apache DataFusion. This project provides graph
processing capabilities on top of DataFusion's DataFrame API.

## About

This project aims to bring the power of GraphFrames to the Apache DataFusion ecosystem by leveraging DataFrame capabilities.
It provides a similar API to Apache Spark's GraphFrames.

## Project Status

The project is in early development stage. Currently implemented features include basic graph operations, statistics,
and Pregel API.

## Features

| Graph Routine                   | GraphFrames | graphframes-rs |
|---------------------------------|-------------|----------------|
| Graph Abstraction               | ✓           | ✓              |
| Basic Statistics (degree, etc.) | ✓           | ✓              |
| Pregel API                      | ✓           | ✓              |
| Shortest Paths                  | ✓           | In Progress    |
| PageRank                        | ✓           | Planned        |
| Parallel Personalized PageRank  | ✓           | Planned        |
| Connected Components            | ✓           | Planned        |
| Strongly Connected Components   | ✓           | Planned        |
| Triangle Count                  | ✓           | Planned        |
| Label Propagation               | ✓           | Planned        |
| Breadth-First Search            | ✓           | Planned        |
| Aggregate Messages              | ✓           | Planned        |
| SVD++                           | ✓           | Planned        |
| Pattern Matching                | ✓           | Planned        |


