#!/bin/bash

echo "Adding all files..."
git add .
echo "Give message"
read message
git commit -m message
echo "Pushing.."
git push
