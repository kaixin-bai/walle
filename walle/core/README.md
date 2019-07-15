# Core

In this module, I'm going to be learning about representing position and orientation in 3-D space. There are 2 amazing repositories that I've been using as black boxes - [math3d](https://github.com/mortlind/pymath3d) and [transformations.py](https://www.lfd.uci.edu/~gohlke/code/transformations.py.html) - so in the spirit of understanding and mastering them, I'm going to be writing my own, simplified library and testing against theirs.

**Motivation.** I've been using the UR5 robot from Universal Robots the past few months. To control the tool center pose (TCP), I've been sending it a 6-element vector `[x, y, z, rx, ry, rz]` where the first three elements correspond to the position of the TCP in Cartesian space, and the last three correspond to its orientation in axis-angle representation. This axis-angle representation has made me lose a few hairs because it's not intuitive to visualize and compare. For example, I've been defining and saving different poses for the robot: one above the receptacle containing the items to be picked and one above the receiving receptacle for the items to be dropped. Position-wise, it's easy to compare these two poses and say, figure out that one is further from the world origin than the other. But orientation-wise, it's been a nightmare comparing and manipulating them. If I try to directly alter one of the last three values to "nudge" the robot arm in a direction I want, the end-effector can go bonkers and reorient itself in a totally unexpected way.

## API

* [Quaternion](https://github.com/kevinzakka/walle/blob/master/walle/core/quaternion.py#L10): for general quaternion algebra.
* [UnitQuaternion](https://github.com/kevinzakka/walle/blob/master/walle/core/quaternion.py#L297): a child class of `Quaternion` useful for representing orientations and rotations in 3-D space.
* [Orientation](https://github.com/kevinzakka/walle/blob/master/walle/core/orientation.py): this class will unify all representations into a single API. It will ingest an orientation specified in any of the 4 representations, perform the required math in quaternion domain, and present the answer in any representation desired by the user.

I'm also adding unit tests for these classes:

- [TestQuaternion](https://github.com/kevinzakka/walle/blob/master/walle/tests/test_quaternion.py)
- [TestUnitQuaternion](https://github.com/kevinzakka/walle/blob/master/walle/tests/test_unit_quaternion.py)
- [TestOrientation](https://github.com/kevinzakka/walle/blob/master/walle/tests/test_orientation.py)

Finally, I'm writing proofs to some of the equations I use [here](https://github.com/kevinzakka/walle/blob/master/proofs/quaternions.pdf).

## Conventions

I'm going to be using a **right-handed** coordinate system.

* Using my right hand, `+X` points to the right of my computer monitor (my thumb), `+Y` points to the top of my monitor (my forefinger) and `+Z` points out of my monitor towards me (my middle finger). This is illustrated in the figure below.
* To find out which direction `+45` degrees rotates, I align the thumb of my right hand with the axis of rotation. Then, the direction my fingers curl is the positive angle.

<p align="center">
 <img src="../../assets/right-hand-rule.png" width="35%">
</p>

The terms **orientation** and **rotation**, whilst related and represented by the same tools, refer to two different things:

* An orientation is a state of the object used to characterize its pose in 3-D space.
* A rotation is an operation carried out on a certain object to modify its orientation.

## Common Vector Operations

* **Subtraction**: The vector `C = B - A`, obtained by subtracting the vector `A` from the vector `B`, represents the direction and distance from `A` to `B`.
* **Addition**: The vector `C = A + B` is obtained by placing the tail of vector `B` on vector `A`'s head and tracing a vector from `A`'s tail to `B`'s head. This is known as triangle completion and can also be done by completing the parallelogram.
* **Scalar Multiplication**: multiplying a vector by a scalar scales the length of the vector but does not alter its direction. A scalar value below 1 contracts the vector and a value above 1 stretches it.
* **Cross Product**: the cross product of two vectors, denoted by the symbol `x`, generates a vector that is orthogonal to the plane defined by the two initial vectors.
* **Dot Product**: the dot product of two vectors, denoted by the symbol `.`, allows us to find the angle between them.

## Why Quaternions?

When dealing with rotations, there are 2 main operations we usually perform: transforming a point (matrix-vector multiplication) and composing rotations (matrix-matrix multiplication). In a nutshell, the first operation is cheaper for matrices and the second is cheaper for unit quaternions.

| Operation            | Rotation Matrix | Unit Quaternion |
|----------------------|-----------------|-----------------|
| Vector Rotation      | 15 (9*, 6+)     | 30 (15*, 15+)   |
| Rotation Composition | 45 (27*, 18+)   | 28 (16*, 12+)   |

But this is not the main reason for using quaternions. Their real forté lies in the fact that they are an "easier" [class invariance](https://stackoverflow.com/a/19583347/4875916): quaternion violations are cheaper to detect and quaternion maintenance is cheaper to enforce. What does this mean exactly?

Let us assume that we represent rotations as `3x3` matrices `R`. For `R` to be a rotation matrix, it must be orthogonal `np.dot(R, R.T) = np.eye(3)` and `det(R) = 1`. The problem is that matrix multiplications -- in the case of rotation chaining -- introduce numerical error which slowly but invariantly accumulates. The result is that these rotation properties get violated and we must (1) detect them by checking orthogonality and (2) fix this "matrix drift" using complex and expensive operations such as SVD decomposition. With quaternions, we must simply check that the norm is unitary to ensure that it represents a rotation. This is a simple check (vector dot product with itself) and enforcing it by normalizing the vector can be done pretty fast using square root approximations. Thus, quaternions are a simpler class invariant than rotation matrices.

Another advantage of quaternions is that going from a unit quaternion to a rotation matrix is cheap and straightforward. The inverse is not and requires multiple edge-case checks. What this means is that if we have a large number of points we would like to rotate, we can easily transform the quaternion to a rotation matrix and perform the multiplication to save computation.

A few more advantages:

* **Storage**: we only need to keep track of 4 numbers for quaternions vs 9 for rotation matrices.
* **Interpolation**: is cheaper and has constant angular velocity which is more aesthetically pleasing.

It is important to mention a small disadvantage of unit quaternions: there isn't a 1-to-1 mapping between rotation matrices and quaternions. A rotation matrix can be represented by two quaternions, `q` and `-q`. This means two things:

* To check whether two quaternions `q1` and `q2` represent the same **orientation**, we need to check that `abs(q1.dot(q2)) > 1 - EPS`. This checks that both `q1 == q2` and `q1 == -q2`.
* To check whether two quaternions `q1` and `q2` represent the same **rotation**, we just need to check that they are both component-wise approximately equal. This is because `q` represents a rotation by `theta` while `-q` represents a rotation by `-theta`.

## Logbook

**04/01/2019**

After a bit of browsing and online reading, I've learned that there are many ways of representing orientation in 3D space, each with its pros and cons:

* Rotation Matrices
* Three-Angles (Euler and Tait-Bryan)
* Axis-Angle
* Unit Quaternions

Three-angles are a nightmare to deal with for many reasons (e.g. gimble lock, complexity of interpolation, imprecision and complexity of composition of rotations, etc.). They are however intuitive to understand and visualize. Rotation matrices are great for a couple reasons: rotation can be represented by a left matrix multiplication and successive rotations correspond to successive matrix multiplications. The downside is that they suffer from numerical drift due to finite-point precision which is costly to renormalize and they constitute a non-minimal description (9 numbers) that can become costly to compute and store. Quaternions are the best of both worlds. As Wikipedia states: "Compared to Euler angles they are simpler to compose and avoid the problem of gimbal lock. Compared to rotation matrices they are more compact, more numerically stable, and more efficient."

So the takeaway is that internally, we should deal with orientation using Unit Quaternions and provide an API that exposes this orientation in its different forms: Euler angles, rotation matrix and axis-angle. So I'm going to start by learning about quaternion algebra and implementing a `Quaternion` class in Python. This will require that I overload the different algebra operators (addition, subtraction, multiplication, etc.) to support complex number manipulation.

**EDIT:** I've found a quote that validates my thinking from an OpenGL [blogpost](http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-17-quaternions/):

> The general consensus is exactly that: use quaternions internally, and expose Euler angles whenever you have some kind of user interface.

**07/01/2019**

Euler's rotation theorem maintains that any rotation or sequence of rotations about a fixed point is equivalent to a single rotation given by an angle theta about a fixed axis called the Euler axis that runs through the fixed point. It is represented by a unit vector `u`. Therefore, any rotation in three dimensions can be represented as a combination of a vector `u` and a scalar `theta` (axis-angle representation).

Quaternions give a convenient way of encoding this axis-angle representation in 4 numbers. Then, to rotate a point represented as a position vector, we create a quaternion `p = [0, point]` with zero scalar part, convert the axis-angle representation to a unit quaternion, and evaluate the conjugation of p by q `p'=qpq^-1` which represents the rotation of the point by theta around the axis `u`. The vector part of the resulting quaternion is the desired vector `p′`.

**09/01/2019**

The classical way of rotating a point about the origin is to pre-multiply the point, represented as a column vector, by a rotation matrix `R`. For every axis in our coordinate system `(x, y, z)`, we have a formula for creating this rotation matrix. For example, a rotation about the z-axis by an angle theta can be represented using the following `3x3` matrix:

```
R_z = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0],
    [0, 0, 1],
])
```

The issue is that this constrains us to rotating about the origin. To rotate about an axis parallel with one of the Cartesian axes, what we can do is find the intersection of the axis with two other axes, temporarily translate the point to be rotated by these offsets, rotate about the origin, and then translate back by an equal and opposite amount. Remember that to represent translation, we need to work with homogeneous coordinates, i.e. `[x, y, z, 1]`. Thus, we can represent the aforementioned trick as a translation, followed by a rotation, followed by an opposite translation. This corresponds to 3 matrix multiplications which we can compose into a single `4x4` matrix. For example, a rotation about an axis parallel to the z-axis that intersects the point `(t_x, t_y, 0)` can be represented by the following `4x4` matrix:

```
R_z = np.array([
    [np.cos(theta), -np.sin(theta), 0, t_x*(1 - np.cos(theta)) + t_y*np.sin(theta)],
    [np.sin(theta), np.cos(theta), 0, t_y*(1 - np.cos(theta)) + t_x*np.sin(theta)],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
])
```

Quick note to myself for future reference. When we rotate about a certain axis, say the z-axis, by any angle theta, the z-component of the point being rotated stays the same. Think about this in 2D, when we rotate in the x-y plane. What's truly happening is that we're rotating about the z-axis but staying in the x-y plane, i.e. z stays 0 the whole time.

OK, so I've seen how to apply a rotation around the origin about any Cartesian axis, and how to translate about an axis parallel to any Cartesian axis. If we want to apply multiple rotations about these different axes, we just need to compose these rotations and perform matrix multiplications. So these single rotations can be combined to produce double and triple transforms. More generally, we can represent a rotation as consecutive rotations about these three elemental Cartesian axes, which we call Euler rotations.

**11/01/2019**

Today, I'm going to try and go through the derivations involving representing quaternions as rotation matrices and efficient, numerically-stable algorithms for converting a rotation matrix into a quaternion. The [Wikipedia page](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation) for quaternion rotations gives two representations for the orthogonal matrix representation of a quaternion: the first time for showing the matrix representation of vector rotation `p' = q * p * q.inv() = Mp` and the second time for converting a quaternion to an orthogonal matrix.

It turns out the 2 matrices provided are equivalent. I've verified this both with code -- by generating multiple random quaternions and comparing both matrices for equality -- and mathematically (see theorem 3 in proof pdf). So we have an easy way of going from quaternion to rotation matrix. It turns out however that the opposite direction needs a bit more work because it is prone to numerical instability.

The first thing we need to take care of is making sure the the rotation matrix we are given is a true rotation matrix. The reason is that when multipliying many rotation matrices, numerical inaccuracies tend to accumulate and the resulting rotation matrix ceases to be truly orthogonal (this is known as matrix drift). This means we need to check for two conditions when given a rotation matrix:

* `rotm @ rotm.T = I`
* `det(rotm) = +1`

If both these conditions aren't satisfied, we need to re-orthonormalize the matrix. There are different ways of carrying out this operation:

* Find the nearest orthogonal matrix using the [Orthogonal Procrustes Problem](https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem). Given a matrix `R`, we can find the nearest orthogonal matrix to `R` by computing the singular value decomposition of `R = UEV^T`. Then the solution is `R' = np.dot(U, V.T)`. I've included the proof to this in Theorem 6. Note that this doesn't guarantee that the rotation matrix is proper.
* Finding the nearest orthogonal matrix using a symmetric [Lagrangian Multiplier Matrix](http://people.csail.mit.edu/bkph/articles/Nearest_Orthonormal_Matrix.pdf). This method also does not guarantee that the rotation matrix is proper. I've implemented this but the resulting matrix is still not proper. The notation in the paper is weird because the trick for computing the inverse of the square root does not produce a matrix but a scalar.
* Re-normalizing the columns of the orthogonal matrix by rotating the `X` and `Y` columns by equal and opposite angles. This angle is calculated from the dot product of `X` and `Y` and is a measure of how much the `X` and `Y` rows are rotating toward each other. This method is described in [Direction Cosine Matrix IMU: Theory](https://wiki.paparazziuav.org/w/images/e/e5/DCMDraft2.pdf).
* QR decomposition of the matrix and setting `R = Q`.
* Gram-Schmidt: this is the classical way of orthogonalizing a matrix. However, we won't implement it since it suffers from numerical instability.

Finally, there's a way of creating a quaternion from a rotation matrix when the matrix is not precise. This involves creating a symmetric matrix from the rotation matrix after which the quaternion is the eigenvector that corresponds to largest eigenvalue. This method is described in Bar-Itzhack's [New Method for Extracting the Quaternion from a Rotation Matrix](https://arc.aiaa.org/doi/abs/10.2514/2.4654?journalCode=jgcd). I unfortunately cannot access this paper because it is behind a paywall.

**15/01/2019**

Today, I'll be finishing the `rotm2quat` function. I need to implement the algorithm described in Mike Day's [pdf](https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf) and the Bar-Itzhack algorithm in case the rotation matrix is not precise.

OK, I'm done implementing `rotm2quat`. I want to revisit it at a later date because I don't feel 100% comfortable with the improvements made to the algorithm by Mike Day.

**17/01/2019**

I'm writing my thought process for the `__mul__` method of the `Orientation` class. Here's what I'm thinking:

* If we're multiplying two `Orientation` objects, then we should multiply their quaternion representations directly.
* If we're multiplying an `Orientation` object and a 3D vector, we should multiply the orientation's quaternion and the vector. The `UnitQuaternion` already deals with this internally and converts the vector to a pure quaternion and performs thr efficient Rodrigues formula for rotation.
* If we're multiplying an `Orientation` object with a batch of 3D vectors, we should pay the price of converting the orientation's quaternion to a rotation matrix and then perform matrix multiplication of the rotation matrix and the batch of vectors. This is because a single rotation of a vector by a quaternion is more efficient in rotation matrix form.
* If we're multiplying an `Orientation` object with a `UnitQuaternion`, we should multiply both quaternion representations directly.
* If we're multiplying an `Orientation` object with a batch of `UnitQuaternion`s, we should chain multiply the quaternion representations directly.

**18/01/2019**

I'm going to try and finish up the `Orientation` class today. I need to add methods that will make it intuitive and easy to perform rotations about different axes. So things like `new_rot_x`, `new_rot_y`, and `new_rot_z` to create rotations about the Cartesian axes.

So I'm trying to implement 2 subproblems today:

* Finding the quaternion that rotates `quat1` to `quat2`.
* Finding the quaternion that rotates `vec1` to `vec2`.

The first problem is pretty easy to solve. We formulate it as trying to find the quaternion `qx` such that `qx * q1 = q2`. The answer is pretty straightforward, i.e. `qx = q2 * q1.inv()`. I've implemented this function as `from_to_quat` in the `UnitQuaternion` class.

The second problem is more involved.

## Q&A With Myself

> Does any completely random quaternion -- normalised to unit-norm -- represent a valid rotation?

In a nutshell yes. Once the quaternion is normalised, it becomes a unit quaternion which can always be expressed in polar form as a cosine plus a sine. Here's some code to test this:

```python
q = np.random.randn(4)  # generate a random quaternion
q /= np.linalg.norm(q)  # normalize it
angle = 2 * np.arccos(s)  # compute rotation
axis = q[1:] / np.sin(angle / 2)  # compute axis
```

The code above runs with a randomly generated quaternion. Note that it could fail for weird edge-cases such as `q` having a near-zero norm or the rotation angle being very near zero in which case `np.sin(0) = 0`.

> Examine the relationship between a quaternion `q`, its negative `-q` and its inverse `q.inv()`.

A unit quaternion `q` and its negative `-q` represent the same rotation matrix.

```python
quat1 = UnitQuaternion.random()
quat2 = -quat1
np.allclose(quat1.rotm, quat2.rotm)  # returns True
```

They represent the same **orientation** in 3D space, but not the same **rotation**. We can check by examining that their axis-angle representations are different.

```python
quat1.axis_angle  # array([-0.89704346,  0.34140664,  0.2806324 ]), 2.4643574821821375
quat2.axis  # array([ 0.89704346, -0.34140664, -0.2806324 ]), 3.818827824997449
```

To be more precise, `-q` represents a **negative** rotation by an axis pointing in the **opposite direction**.

```python
axis, angle = quat1.axis_angle
expected_neg_axis = -axis
expected_neg_angle = 2*np.pi + (-angle)
actual_neg_axis, actual_neg_angle = quat2.axis_angle
np.allclose(actual_neg_axis, expected_neg_axis)  # returns True
np.isclose(actual_neg_angle, expected_neg_angle)  # returns True
```

How about `q` and `q.inv()`? Well it turns out that the inverse of a unit quaternion is just its conjugate, which means it's the same quaternion but its vectorial part is negated. If we think about this in axis-angle representation, then the unit-vector has been negated, i.e. it now points in the opposite direction. Thus, the inverse of a quaternion rotates by the same number of degrees, but in the opposite direction. This is equivalent to reversing the direction of rotation. The conclusion is that `q.inv()` cancels out the rotation done by `q`.

```python
quat1.axis_angle  # array([-0.89704346,  0.34140664,  0.2806324 ]), 2.4643574821821375
quat3.axis_angle  # array([ 0.89704346, -0.34140664, -0.2806324 ]), 2.4643574821821375
```

As we can see above, we have the same angle of rotation, but the axis of rotation points in the opposite direction. Finally, we examine its effect on a vector:

```python
vec = np.array([-5, 3, 2])
rotated_vec = quat1 * vec  # [-5.89269679, 1.59479987, 0.8560011]
np.allclose(vec, quat3 * rotated_vec)  # returns True
```

> Computing the quaternion that rotates a vector to another is not the same as treating the vectors as pure quaternions and computing the rotation quaternion.

Add explanation here.

## Todos

**Short Term.**

- [ ] Write unit tests for `orientation`, `matrix`, `quaternion` and `pose`.
- [ ] Learn correct way of dealing with machine epsilon.
    - [ ] Fix `np.isclose(np.dot(u, u), 1.)` in axis-angle.
- [x] Implement `Pose` class to consolidate position and orientation classes.
- [x] Figure out how to deal with left or right multiplication of `Orientation` objects.
- [x] Debug `from_vecs` method in `UnitQuaternion` class.
- [x] Add edge-case support for `slerp`.
- [x] Add batch of vector support to `UnitQuaternion` multiplication.
- [x] Add random sampling of rotation matrices and quaternions.
- [x] Add inplace operation support.
- [x] Implement spherical linear interpolation (slerp).

**Long Term.**

- [ ] Add Taylor Expansions for more precise edge-case calculations.
- [ ] Current `slerp` isn't efficient. Compare with Dantam's version.

## Notes

- I decided to go with matrix multiplication for single vector or batch of vectors instead of quaternion multiplication for single vector case. This makes the code more modular.
- In the UnitQuaternion `__mul__` method, in the case where the operand is a pure quaternion, we return a `Quaternion` object rather than a vector because the user has given to us as input a Quaternion. Therefore, we cannot assume that the intent was to rotate the vector stored in the Quaternion's vectorial part. This means the user must explicitly extract the rotated position vector from the resulting quaternion.
- The default constructor for a `UnitQuaternion` returns an identity quaternion which represents no rotation. Its axis angle representation has `theta = 0` which means there are an infinite number of valid axes. I've decided to return the Cartesian `i = [1, 0, 0]` axis for any such null rotation.
- I chose not to implement addition and subtraction operations for unit quaternions. The reason is that deciding what the user wants is ambiguous. Does the user want to add/subtract two quaternions and obtain the normalized resulting quaternion? Or does the user just want to see the raw `q` vector resulting from addition/subtraction? This line of questioning is valid because adding/subtracting unit quaternions is not closed, i.e. it does not guarantee that the result is a unit-quaternion unless explicit normalization is performed.

## References

- [Understanding Quaternions](https://www.3dgep.com/understanding-quaternions/#The_Complex_Plane)
- [Quaternions and Rotations](http://graphics.stanford.edu/courses/cs348a-17-winter/Papers/quaternion.pdf)
- [Wikipedia 1](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation)
- [Wikipedia 2](https://en.wikipedia.org/wiki/Quaternion)
- [Code Example for Quaternion](https://stackoverflow.com/questions/4870393/rotating-coordinate-system-via-a-quaternion)
- [Notes on Quaternions](https://users.aalto.fi/~ssarkka/pub/quat.pdf)
- [Pros and Cons](https://math.stackexchange.com/questions/1355081/why-is-representing-rotations-through-quaternions-more-compact-and-quicker-than)
- [Quaternions for Computer Graphics](https://www.springer.com/gp/book/9780857297594)
- [Generating a random element of `SO(3)`](http://planning.cs.uiuc.edu/node198.html)
- [OpenGL Tutorial 17](http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-17-quaternions/)
- [Ogre Rotation Primer](http://wiki.ogre3d.org/Quaternion+and+Rotation+Primer)
- [Fast Quaternion Normalization](https://stackoverflow.com/questions/11667783/quaternion-and-normalization/12934750#12934750)
- [Comparison](https://math.stackexchange.com/questions/1386003/what-are-advantages-of-quaternion-over-3-times3-rotator-matrix-for-representin)
- [Comparing Quaternions](https://gamedev.stackexchange.com/a/75108)
- [Precise Quaternion Calculations With Taylor Expansions](http://www.neil.dantam.name/note/dantam-quaternion.pdf)
- [Converting a Rotation Matrix to a Quaternion](https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf)
- [Fast Inverse Square Root](https://en.wikipedia.org/wiki/Fast_inverse_square_root)
- [Quaternion Report](http://web.mit.edu/2.998/www/QuaternionReport1.pdf)
- [Generate an Orthogonal Vector to a Random Input Vector](http://lolengine.net/blog/2013/09/21/picking-orthogonal-vector-combing-coconuts)
- [Quaternion from 2 Vectors](http://lolengine.net/blog/2014/02/24/quaternion-from-two-vectors-final)