<p align="center">
  <a href="https://gladia.io"><img src="https://i.ibb.co/BgD6s0x/Gladia.png" alt="Gladia" width="100%"></a>
</p>
<p align="center">
  <em>Gladia the fastest way to develop and deploy AI APIs.</em>
  <br/>
  <em>The fastest way to implement and compare AI research models for a given task.</em>
</p>

---
**Open Source Project Website**: <a href="https://gladia.io/" target="_blank">https://gladia.io/</a>

**Documentation**: <a href="https://docs.gladia.io/" target="_blank">https://docs.gladia.io/</a>

**Documentation Source Code**: <a href="https://github.com/gladiaio/docs/" target="_blank">https://github.com/gladiaio/docs</a>

**Framework Source Code**: <a href="https://github.com/gladiaio/gladia/" target="_blank">https://github.com/gladiaio/gladia</a>

**Discord**: <a href="https://discord.gg/HeuCTxnXrE" target="_blank">https://discord.com/invite/HeuCTxnXrE</a>

---
<p align="left">
  <a href="https://github.com/gladiaio/gladia/issues" alt="Issues">
    <img src="https://img.shields.io/github/issues/gladiaio/gladia" />
  </a>
  <a href="https://github.com/gladiaio/gladia/pulls" alt="Pull Requests">
    <img src="https://img.shields.io/github/issues-pr/gladiaio/gladia" />
  </a>
  <a href="https://github.com/gladiaio/gladia/network/members" alt="Forks">
    <img src="https://img.shields.io/github/forks/gladiaio/gladia" />
  </a>
  <a href="https://github.com/gladiaio/gladia/stargazers" alt="Stars">
    <img src="https://img.shields.io/github/stars/gladiaio/gladia" />
  </a>
  <a href="https://results.pre-commit.ci/latest/github/gladiaio/gladia/main" alt="pre-commit.ci">
    <img src="https://results.pre-commit.ci/badge/github/gladiaio/gladia/main.svg" />
  </a>
  <a href="https://github.com/psf/black" alt="code-style">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" />
  </a>
  <a href="https://sonarcloud.io/project/overview?id=gladiaio_gladia" alt="code-quality">
    <img src="https://sonarcloud.io/api/project_badges/measure?project=gladiaio_gladia&metric=alert_status" />
  </a>
  <a href="https://hub.docker.com/repository/docker/gladiaio/gladia" alt="code-quality">
    <img src="https://img.shields.io/docker/pulls/gladiaio/gladia.svg" />
  </a>
  <a href="https://github.com/gladiaio/gladia/actions/workflows/gladia-latest-deploy.yml" alt="code-quality">
    <img src="https://github.com/gladiaio/gladia/actions/workflows/gladia-latest-deploy.yml/badge.svg" />
  </a>
</p>


[![](https://dcbadge.vercel.app/api/server/UUd79ckzz9)](https://discord.gg/UUd79ckzz9)

Follow Gladia Twitter account to get the latest update
<p align="left">
  <a href="https://twitter.com/gladia_io" alt="Twitter">
    <img src="https://img.shields.io/twitter/follow/gladia_io.svg?style=social&label=Follow Gladia" />
  </a>
</p>

[Follow Gladia Twitter account to get the latest update](https://twitter.com/gladia_io)

[Join our Discord Community Server](https://discord.com/invite/HeuCTxnXrE)


# What's Gladia
## Why Gladia
AI is not an easy topic, we have solved that for you! The web developer 1 stop shop for AI APIs. Translation, Speech to Text and Text to Speech (TTS), Sentiment Analysis, Voice synthesis (speech-to-text), Offensive Content Detection, Background removal, Face Detection, Object Detection, and more to come. One library to rule them all!
## Why you should contribute ?
Help us build the biggest treasure trove of State of the Arts AI models, one we all can benefit from.


# Quickstart
## :rocket: Instant Start

When it comes to get hands on a product to test, nothing's better than a one liner ! Here is the magic: 

```sh
$ docker run -d --gpus all --shm-size=5g -p 8080:8080 gladiaio/gladia:latest
```

Your Gladia service is now available at [http://localhost:8080/docs](http://localhost:8080/docs)


## :mechanical_arm: Full Start 

if you want to get a taste of Gladia full power, use it with the integrations !

For advanced AI Image features, Gladia integrates with HuggingFace and DreamStudio API
  
HuggingFace is a reference when it comes to models sharing. Its hub is massive and helps AI community to build and share models on a daily basis. You will need a `HUGGING_FACE_ACCESS_TOKEN`, just proceed as follows:
1. Create your free account https://huggingface.co/
2. Access https://huggingface.co/settings/tokens
3. Create a `READ` token  
  
DreamStudio is a tool suite namely providing stable diffusion features. They build great stuff and contribute to the AI creation ecosystem.
You will also need an API key, it is available in only 2 steps:
1. Create your free account https://beta.dreamstudio.ai/
2. Get your token on https://beta.dreamstudio.ai/membership

You're ready to go !  


```sh
# to test image generation for both stability API and the open source stable diffusion
# you'll need STABILITY_KEY and HUGGINGFACE_ACCESS_TOKEN
# getting STABILITY_KEY here https://beta.dreamstudio.ai/membership
# getting HUGGINGFACE_ACCESS_TOKEN here https://huggingface.co/settings/tokens

$ docker run -d --gpus all --shm-size=5g -p 8080:8080 -e STABILITY_KEY=sk-***** -e HUGGINGFACE_ACCESS_TOKEN=hf_***** gladiaio/gladia:latest

```
Access the service through [http://localhost:8080/docs](http://localhost:8080/docs) or [http://localhost:8080/redoc](http://localhost:8080/redoc) or whatever public/private IP of the server you are running on

/!\ The First API call of each endpoint might/will be slower as its preforming lazy model caching (after the first call should be ok).

## What's behind the Quickstart scene
```sh
git clone https://github.com/gladiaio/gladia.git
cd gladia/src

#building the Docker Image
docker build -t gladia -f gpu.Dockerfile .
```

## Running the dev environement
This will create a 'model' volume that will store, once and for all, the AI models
```sh
docker-compose down && \
docker-compose pull && \
docker-compose -f docker-compose.yaml -f docker-compose.dev.yaml up -d
```

You can stop the services without removing the 'model' volume
```sh
docker-compose down
```

## Running the production environement
This will create a 'model' volume that will store, once and for all, the AI models
```sh
docker-compose down && \
docker-compose pull && \
docker-compose -f docker-compose.yaml -f docker-compose.prod.yaml up -d
```

You can stop the services without removing the 'model' volume
```sh
docker-compose down
```

You can log in one the instances and run the tests 
```sh
docker-compose exec gladia /bin/bash
cd unit-test
python test.py -c
```

To know more about test please check this [readme dedicated to test](https://github.com/gladiaio/gladia/blob/main/src/tests/README.md)

## As a new developer, very first steps in full control

We created a [dedicated page](./src/howto101.md) with detailed instructions on how, as a new develper, to perform few first steps actions. The objective is for the new developer to be in full control while performing the following first steps :

1. To be able to build and run your own image based on the repository sources
2. Make a first API call to your own newly running container
3. Run the test suite within the newly running container


# Project Management
- [core framework](https://github.com/gladiaio/gladia/projects/1)
- [AI API collection](https://github.com/gladiaio/gladia/projects/2)
- [Issues](https://github.com/gladiaio/gladia/projects/3)
- [Pull Requests](https://github.com/gladiaio/gladia/projects/5)
