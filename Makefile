# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* business_case_869/*.py

black:
	@black scripts/* business_case_869/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr business_case_869-*.dist-info
	@rm -fr business_case_869.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)


# project id - replace with your GCP project id
PROJECT_ID=wagon-businesscase-869

# bucket name - replace with your GCP bucket name
BUCKET_NAME=business-case

##### Training  - - - - - - - - - - - - - - - - - - - - - -
# will store the packages uploaded to GCP for the training
BUCKET_TRAINING_FOLDER = 'trainings'

# choose your region from https://cloud.google.com/storage/docs/locations#available_locations
REGION=southamerica-east1

set_project:
	@gcloud config set project ${PROJECT_ID}

create_bucket:
	@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}


##### Job - - - - - - - - - - - - - - - - - - - - - - - - -
JOB_NAME=business_case_869_$(shell date +'%Y%m%d_%H%M%S')

##### Package params  - - - - - - - - - - - - - - - - - - -
PACKAGE_NAME=package
FILENAME=etl

##### Machine configuration (defaul for GCP) - - - - - - - - - - - - - - - -
PYTHON_VERSION=3.7
FRAMEWORK=scikit-learn
RUNTIME_VERSION=1.15

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
			--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER}  \
			--package-path ${PACKAGE_NAME} \
			--module-name ${PACKAGE_NAME}.${FILENAME} \
			--python-version=${PYTHON_VERSION} \
			--runtime-version=${RUNTIME_VERSION} \
			--region ${REGION} \
			--stream-logs



# path to the file to upload to GCP (the path to the file should be absolute or should match the directory where the make command is ran)
# replace with your local path to the `train_1k.csv` and make sure to put the path between quotes
LOCAL_PATH="model.joblib"

# bucket directory in which to store the uploaded file (`data` is an arbitrary name that we choose to use)
BUCKET_FOLDER='trainings_local'

# name for the uploaded file inside of the bucket (we choose not to rename the file that we upload)
BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH})

upload_data: # to save models traned locally to GCP
  # @gsutil cp train_1k.csv gs://wagon-ml-my-bucket-name/data/train_1k.csv
	@gsutil cp ${LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}
