#!/bin/bash

LOT_NAME=${1:-"lot1"}
SIZE=${2:-"1365"}
OFFER_ID="25522320"

echo "Setting up COLMAP for lot: $LOT_NAME with size: $SIZE"
echo "----------------------------------------"

DATA_DIR="dataset/colmap/${LOT_NAME}/${SIZE}px"
if [ ! -d "$DATA_DIR" ]; then
  echo "Directory not found: $DATA_DIR"
  echo "Please run preprocessing first: python -m dataset.preprocessing"
  exit 1
fi

echo "Building Docker image..."
docker build -t nerfs-arch/colmap:latest -f Dockerfile.colmap . &

echo "Creating data bundle..."
cd dataset/colmap
tar -czf /tmp/${LOT_NAME}_${SIZE}.tar.gz ${LOT_NAME}/${SIZE}px/
cd ../..
wait

echo "Launching vast.ai job..."
cat > /tmp/onstart.sh << 'EOF'
#!/bin/bash
echo "Instance started - setting up..."
sleep 10
EOF

INSTANCE_INFO=$(uv run vastai create instance $OFFER_ID \
    --image nerfs-arch/colmap:latest \
    --disk 80 \
    --onstart /tmp/onstart.sh \
    --env "LOT_NAME=$LOT_NAME" \
    --env "SIZE=$SIZE" \
    --ssh)

echo "Instance creation response:"
echo "$INSTANCE_INFO"

INSTANCE_ID=$(echo "$INSTANCE_INFO" | grep -o '[0-9]\+' | head -1)

if [ -z "$INSTANCE_ID" ] || [ ${#INSTANCE_ID} -lt 3 ]; then
  echo "Could not extract instance ID"
  INSTANCE_ID=$(uv run vastai show instances --raw | grep -o '"id": [0-9]*' | head -1 | cut -d' ' -f2)
fi

if [ -z "$INSTANCE_ID" ]; then
  echo "Still no instance ID. Please check:"
  echo "   - Offer ID: $OFFER_ID"
  echo "   - API key: \$VAST_AI_API_KEY"
  echo "   - Run: uv run vastai search offers 'id=25522320'"
  exit 1
fi

echo "Instance created: $INSTANCE_ID"
echo "Full instance address: root@$INSTANCE_ID.inst.vast.ai"

echo "Waiting for instance to be ready..."
sleep 45

echo "Uploading data bundle to instance..."
scp /tmp/${LOT_NAME}_${SIZE}.tar.gz root@$INSTANCE_ID.inst.vast.ai:/workspace/

echo "Uploading colmap_manager_cloud.py..."
scp dataset/colmap_manager_cloud.py root@$INSTANCE_ID.inst.vast.ai:/workspace/

echo "Running COLMAP on instance..."
ssh root@$INSTANCE_ID.inst.vast.ai << 'EOF'
  cd /workspace
  echo "Creating dataset directory..."
  mkdir -p dataset
  echo "Extracting data to /workspace/dataset/colmap/..."
  tar -xzf ${LOT_NAME}_${SIZE}.tar.gz -C dataset/
  echo "Running COLMAP..."
  python colmap_manager_cloud.py
  echo "COLMAP completed!"
EOF

echo "Downloading results..."
scp -r root@$INSTANCE_ID.inst.vast.ai:/workspace/dataset/colmap/${LOT_NAME}/${SIZE}px/output/ ${DATA_DIR}/

echo "Stopping instance..."
uv run vastai destroy instance $INSTANCE_ID

echo "Cleaning up..."
rm /tmp/${LOT_NAME}_${SIZE}.tar.gz /tmp/onstart.sh

echo "Deployment completed!"
echo "Results: $DATA_DIR/output/"