#!/bin/bash

gcloud run deploy "$SERVICE_NAME"   --port=8080   --image="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME"   --allow-unauthenticated   --region=$GCP_REGION   --platform=managed    --project=$GCP_PROJECT   --set-env-vars=GCP_PROJECT=$GCP_PROJECT,GCP_REGION=$GCP_REGION
