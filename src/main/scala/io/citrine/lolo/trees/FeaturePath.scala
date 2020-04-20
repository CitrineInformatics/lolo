package io.citrine.lolo.trees

/**
  * Description of a feature's role in computing TreeSHAP
  *
  * @param featureIndex index of the feature this node describes.
  * @param zeroFraction fraction of paths flowing through this node with this feature excluded.
  * @param oneFraction fraction of one paths flowing through this node with this feature included.
  * @param pathWeight summation weight for this path, taking into account combinatorial factor.
  */
class FeatureNode(
                  var featureIndex: Int,
                  var zeroFraction: Double,
                  var oneFraction: Double,
                  var pathWeight: Double
                 ) {
  def copy(): FeatureNode = {
    new FeatureNode(featureIndex, zeroFraction, oneFraction, pathWeight)
  }
}

/**
  * Path of unique features used in splitting to arrive at a node in a decision tree
  *
  * Note: FeaturePath.length is the number of splitting features. This does NOT include the first
  * node, which accounts for the "empty path" in computing TreeSHAP.
  *
  * @param numFeatures number of features in the input space.
  */
class FeaturePath(numFeatures: Int) {
  var path: Array[FeatureNode] = Array.fill[FeatureNode](numFeatures + 2)(new FeatureNode(-1, 1, 1, 1))
  var length: Int = -1  // Start at -1 since first path extension accounts for 0 active features.

  /**
    * Extend the path of unique features used in splitting.
    *
    * @param zeroFraction fraction of paths flowing through this node with this feature excluded.
    * @param oneFraction fraction of one paths flowing through this node with this feature included.
    * @param featureIndex index of feature used in this split with which to extend the path.
    * @return this.
    */
  def extend(
             zeroFraction: Double,
             oneFraction: Double,
             featureIndex: Int
            ): FeaturePath = {
    length += 1
    path(length).zeroFraction = zeroFraction
    path(length).oneFraction = oneFraction
    path(length).featureIndex = featureIndex
    path(length).pathWeight = if (length == 0) 1.0 else 0.0

    (length-1 to 0 by -1).foreach{i =>
      path(i + 1).pathWeight += oneFraction * path(i).pathWeight * ((i + 1).toDouble/(length + 1))
      path(i).pathWeight = zeroFraction * path(i).pathWeight * ((length - i).toDouble/(length + 1))
    }

    this
  }

  /**
    * Undo a previous extension of the feature path.
    *
    * @param featureIndex index within the path to unwind.
    * @return unwound copy of this path.
    */
  def unwind(featureIndex: Int): FeaturePath = {
    val out = this.copy()
    val newPath = out.path
    var n = newPath(length).pathWeight

    (length-1 to 0 by -1).foreach{j=>
      if (newPath(featureIndex).oneFraction != 0.0) {
        val t = newPath(j).pathWeight
        newPath(j).pathWeight = n*(length + 1)/((j + 1)*newPath(featureIndex).oneFraction)
        n = t - newPath(j).pathWeight * newPath(featureIndex).zeroFraction * ((length - j).toDouble/(length + 1))
      } else {
        newPath(j).pathWeight = newPath(j).pathWeight*(length + 1).toDouble/(newPath(featureIndex).zeroFraction*(length - j))
      }
    }

    (featureIndex until length).foreach{ i=>
      newPath(i).featureIndex = newPath(i + 1).featureIndex
      newPath(i).zeroFraction = newPath(i + 1).zeroFraction
      newPath(i).oneFraction = newPath(i + 1).oneFraction
    }

    out.length -= 1

    out
  }

  def copy(): FeaturePath = {
    val newPath = new FeaturePath(this.numFeatures)
    newPath.length = this.length
    newPath.path = this.path.map{x => x.copy()}
    newPath
  }

}
