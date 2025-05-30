FROM nvcr.io/nvidia/pytorch:25.03-py3

# Install required packages
#COPY nvidia_resiliency_ext-0.4.0-cp312-cp312-manylinux_2_35_x86_64.whl /tmp/
#RUN pip install /tmp/nvidia_resiliency_ext-0.4.0-cp312-cp312-manylinux_2_35_x86_64.whl
RUN pip install nvidia-resiliency-ext

# Copy test script
COPY test_rendezvous.py /app/
WORKDIR /app

ENV PYTHONPATH=/app

# Set entrypoint
ENTRYPOINT ["python3", "test_rendezvous.py"]
