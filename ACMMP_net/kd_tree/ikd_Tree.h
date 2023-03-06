#pragma once
// #include <pcl/point_types.h>
#include <Eigen/StdVector>
#include <Eigen/Geometry>
#include <stdio.h>
#include <queue>
#include <pthread.h>
#include <chrono>
#include <time.h>
#include <iostream>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include <stdexcept>

// #include <boost/shared_ptr.hpp>



#define EPSS 1e-6
#define Minimal_Unbalanced_Tree_Size 10
#define Multi_Thread_Rebuild_Point_Num 1500
#define DOWNSAMPLE_SWITCH true
#define ForceRebuildPercentage 0.2
#define Q_LEN 1000000

using namespace std;

typedef Eigen::Map<Eigen::Array3f> Array3fMap;
typedef const Eigen::Map<const Eigen::Array3f> Array3fMapConst;
typedef Eigen::Map<Eigen::Array4f, Eigen::Aligned> Array4fMap;
typedef const Eigen::Map<const Eigen::Array4f, Eigen::Aligned> Array4fMapConst;
typedef Eigen::Map<Eigen::Vector3f> Vector3fMap;
typedef const Eigen::Map<const Eigen::Vector3f> Vector3fMapConst;
typedef Eigen::Map<Eigen::Vector4f, Eigen::Aligned> Vector4fMap;
typedef const Eigen::Map<const Eigen::Vector4f, Eigen::Aligned> Vector4fMapConst;

typedef Eigen::Matrix<uint8_t, 3, 1> Vector3c;
typedef Eigen::Map<Vector3c> Vector3cMap;
typedef const Eigen::Map<const Vector3c> Vector3cMapConst;
typedef Eigen::Matrix<uint8_t, 4, 1> Vector4c;
typedef Eigen::Map<Vector4c, Eigen::Aligned> Vector4cMap;
typedef const Eigen::Map<const Vector4c, Eigen::Aligned> Vector4cMapConst;

#define PCL_ADD_UNION_POINT4D \
    union EIGEN_ALIGN16       \
    {                         \
        float data[4];        \
        struct                \
        {                     \
            float x;          \
            float y;          \
            float z;          \
        };                    \
    };

#define PCL_ADD_UNION_NORMAL4D \
    union EIGEN_ALIGN16        \
    {                          \
        float data_n[4];       \
        float normal[3];       \
        struct                 \
        {                      \
            float normal_x;    \
            float normal_y;    \
            float normal_z;    \
        };                     \
    };

#define PCL_ADD_UNION_RGB  \
    union                  \
    {                      \
        union              \
        {                  \
            struct         \
            {              \
                uint8_t b; \
                uint8_t g; \
                uint8_t r; \
                uint8_t a; \
            };             \
            float rgb;     \
        };                 \
        uint32_t rgba;     \
    };

#define PCL_ADD_EIGEN_MAPS_POINT4D                                                                \
    inline Vector3fMap getVector3fMap() { return (Vector3fMap(data)); }                 \
    inline Vector3fMapConst getVector3fMap() const { return (Vector3fMapConst(data)); } \
    inline Vector4fMap getVector4fMap() { return (Vector4fMap(data)); }                 \
    inline Vector4fMapConst getVector4fMap() const { return (Vector4fMapConst(data)); } \
    inline Array3fMap getArray3fMap() { return (Array3fMap(data)); }                    \
    inline Array3fMapConst getArray3fMap() const { return (Array3fMapConst(data)); }    \
    inline Array4fMap getArray4fMap() { return (Array4fMap(data)); }                    \
    inline Array4fMapConst getArray4fMap() const { return (Array4fMapConst(data)); }

#define PCL_ADD_EIGEN_MAPS_NORMAL4D                                                                       \
    inline Vector3fMap getNormalVector3fMap() { return (Vector3fMap(data_n)); }                 \
    inline Vector3fMapConst getNormalVector3fMap() const { return (Vector3fMapConst(data_n)); } \
    inline Vector4fMap getNormalVector4fMap() { return (Vector4fMap(data_n)); }                 \
    inline Vector4fMapConst getNormalVector4fMap() const { return (Vector4fMapConst(data_n)); }

#define PCL_ADD_EIGEN_MAPS_RGB                                                                                                           \
    inline Eigen::Vector3i getRGBVector3i() { return (Eigen::Vector3i(r, g, b)); }                                                       \
    inline const Eigen::Vector3i getRGBVector3i() const { return (Eigen::Vector3i(r, g, b)); }                                           \
    inline Eigen::Vector4i getRGBVector4i() { return (Eigen::Vector4i(r, g, b, a)); }                                                    \
    inline const Eigen::Vector4i getRGBVector4i() const { return (Eigen::Vector4i(r, g, b, a)); }                                        \
    inline Eigen::Vector4i getRGBAVector4i() { return (Eigen::Vector4i(r, g, b, a)); }                                                   \
    inline const Eigen::Vector4i getRGBAVector4i() const { return (Eigen::Vector4i(r, g, b, a)); }                                       \
    inline Vector3cMap getBGRVector3cMap() { return (Vector3cMap(reinterpret_cast<uint8_t *>(&rgba))); }                       \
    inline Vector3cMapConst getBGRVector3cMap() const { return (Vector3cMapConst(reinterpret_cast<const uint8_t *>(&rgba))); } \
    inline Vector4cMap getBGRAVector4cMap() { return (Vector4cMap(reinterpret_cast<uint8_t *>(&rgba))); }                      \
    inline Vector4cMapConst getBGRAVector4cMap() const { return (Vector4cMapConst(reinterpret_cast<const uint8_t *>(&rgba))); }

#define PCL_ADD_POINT4D   \
    PCL_ADD_UNION_POINT4D \
    PCL_ADD_EIGEN_MAPS_POINT4D

#define PCL_ADD_NORMAL4D   \
    PCL_ADD_UNION_NORMAL4D \
    PCL_ADD_EIGEN_MAPS_NORMAL4D

#define PCL_ADD_RGB   \
    PCL_ADD_UNION_RGB \
    PCL_ADD_EIGEN_MAPS_RGB

struct EIGEN_ALIGN16 _PointXYZINormal
{
    PCL_ADD_POINT4D;  // This adds the members x,y,z which can also be accessed using the point (which is float[4])
    PCL_ADD_NORMAL4D; // This adds the member normal[3] which can also be accessed using the point (which is float[4])
    union
    {
        struct
        {
            float intensity;
            float curvature;
        };
        float data_c[4];
    };
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct EIGEN_ALIGN16 _PointXYZRGBA
{
    PCL_ADD_POINT4D; // This adds the members x,y,z which can also be accessed using the point (which is float[4])
    PCL_ADD_RGB;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct PointXYZINormal : public _PointXYZINormal
{
    inline PointXYZINormal(const _PointXYZINormal &p)
    {
        x = p.x;
        y = p.y;
        z = p.z;
        data[3] = 1.0f;
        normal_x = p.normal_x;
        normal_y = p.normal_y;
        normal_z = p.normal_z;
        data_n[3] = 0.0f;
        curvature = p.curvature;
        intensity = p.intensity;
    }

    inline PointXYZINormal()
    {
        x = y = z = 0.0f;
        data[3] = 1.0f;
        normal_x = normal_y = normal_z = data_n[3] = 0.0f;
        intensity = 0.0f;
        curvature = 0;
    }

    friend std::ostream &operator<<(std::ostream &os, const PointXYZINormal &p);
};

struct EIGEN_ALIGN16 PointXYZRGBA : public _PointXYZRGBA
{
    inline PointXYZRGBA(const _PointXYZRGBA &p)
    {
        x = p.x;
        y = p.y;
        z = p.z;
        data[3] = 1.0f;
        rgba = p.rgba;
    }

    inline PointXYZRGBA()
    {
        x = y = z = 0.0f;
        data[3] = 1.0f;
        r = g = b = 0;
        a = 255;
    }

    friend std::ostream &operator<<(std::ostream &os, const PointXYZRGBA &p);
};

struct PCLHeader
{
    PCLHeader() : seq(0), stamp(), frame_id()
    {
    }

    /** \brief Sequence number */
    uint32_t seq;
    /** \brief A timestamp associated with the time when the data was acquired
     *
     * The value represents microseconds since 1970-01-01 00:00:00 (the UNIX epoch).
     */
    uint64_t stamp;
    /** \brief Coordinate frame ID */
    std::string frame_id;

    typedef std::shared_ptr<PCLHeader> Ptr;
    typedef std::shared_ptr<PCLHeader const> ConstPtr;
}; // struct PCLHeader

struct FieldMapping
{
    size_t serialized_offset;
    size_t struct_offset;
    size_t size;
};
typedef std::vector<FieldMapping> MsgFieldMap;
template <typename PointT> class PointCloud;
template <typename PointT> std::shared_ptr<MsgFieldMap>& getMapping (PointCloud<PointT>& p);

template <typename PointT>
class PointCloud
{
public:
    /** \brief Default constructor. Sets \ref is_dense to true, \ref width
     * and \ref height to 0, and the \ref sensor_origin_ and \ref
     * sensor_orientation_ to identity.
     */
    PointCloud() : header(), points(), width(0), height(0), is_dense(true),
                   sensor_origin_(Eigen::Vector4f::Zero()), sensor_orientation_(Eigen::Quaternionf::Identity()),
                   mapping_()
    {
    }

    /** \brief Copy constructor (needed by compilers such as Intel C++)
     * \param[in] pc the cloud to copy into this
     */
    PointCloud(PointCloud<PointT> &pc) : header(), points(), width(0), height(0), is_dense(true),
                                         sensor_origin_(Eigen::Vector4f::Zero()), sensor_orientation_(Eigen::Quaternionf::Identity()),
                                         mapping_()
    {
        *this = pc;
    }

    /** \brief Copy constructor (needed by compilers such as Intel C++)
     * \param[in] pc the cloud to copy into this
     */
    PointCloud(const PointCloud<PointT> &pc) : header(), points(), width(0), height(0), is_dense(true),
                                               sensor_origin_(Eigen::Vector4f::Zero()), sensor_orientation_(Eigen::Quaternionf::Identity()),
                                               mapping_()
    {
        *this = pc;
    }

    /** \brief Copy constructor from point cloud subset
     * \param[in] pc the cloud to copy into this
     * \param[in] indices the subset to copy
     */
    PointCloud(const PointCloud<PointT> &pc,
               const std::vector<int> &indices) : header(pc.header), points(indices.size()), width(indices.size()), height(1), is_dense(pc.is_dense),
                                                  sensor_origin_(pc.sensor_origin_), sensor_orientation_(pc.sensor_orientation_),
                                                  mapping_()
    {
        // Copy the obvious
        assert(indices.size() <= pc.size());
        for (size_t i = 0; i < indices.size(); i++)
            points[i] = pc.points[indices[i]];
    }

    /** \brief Allocate constructor from point cloud subset
     * \param[in] width_ the cloud width
     * \param[in] height_ the cloud height
     * \param[in] value_ default value
     */
    PointCloud(uint32_t width_, uint32_t height_, const PointT &value_ = PointT())
        : header(), points(width_ * height_, value_), width(width_), height(height_), is_dense(true), sensor_origin_(Eigen::Vector4f::Zero()), sensor_orientation_(Eigen::Quaternionf::Identity()), mapping_()
    {
    }

    /** \brief Destructor. */
    virtual ~PointCloud() {}

    /** \brief Add a point cloud to the current cloud.
     * \param[in] rhs the cloud to add to the current cloud
     * \return the new cloud as a concatenation of the current cloud and the new given cloud
     */
    inline PointCloud &
    operator+=(const PointCloud &rhs)
    {
        // Make the resultant point cloud take the newest stamp
        if (rhs.header.stamp > header.stamp)
            header.stamp = rhs.header.stamp;

        size_t nr_points = points.size();
        points.resize(nr_points + rhs.points.size());
        for (size_t i = nr_points; i < points.size(); ++i)
            points[i] = rhs.points[i - nr_points];

        width = static_cast<uint32_t>(points.size());
        height = 1;
        if (rhs.is_dense && is_dense)
            is_dense = true;
        else
            is_dense = false;
        return (*this);
    }

    /** \brief Add a point cloud to another cloud.
     * \param[in] rhs the cloud to add to the current cloud
     * \return the new cloud as a concatenation of the current cloud and the new given cloud
     */
    inline const PointCloud
    operator+(const PointCloud &rhs)
    {
        return (PointCloud(*this) += rhs);
    }

    /** \brief Obtain the point given by the (column, row) coordinates. Only works on organized
     * datasets (those that have height != 1).
     * \param[in] column the column coordinate
     * \param[in] row the row coordinate
     */
    inline const PointT &
    at(int column, int row) const
    {
        if (this->height > 1)
            return (points.at(row * this->width + column));
        else
            // throw IsNotDenseException("Can't use 2D indexing with a unorganized point cloud");
            throw std::invalid_argument("Can't use 2D indexing with a unorganized point cloud");
    }

    /** \brief Obtain the point given by the (column, row) coordinates. Only works on organized
     * datasets (those that have height != 1).
     * \param[in] column the column coordinate
     * \param[in] row the row coordinate
     */
    inline PointT &
    at(int column, int row)
    {
        if (this->height > 1)
            return (points.at(row * this->width + column));
        else
            // throw IsNotDenseException("Can't use 2D indexing with a unorganized point cloud");
            throw std::invalid_argument("Can't use 2D indexing with a unorganized point cloud");

    }

    /** \brief Obtain the point given by the (column, row) coordinates. Only works on organized
     * datasets (those that have height != 1).
     * \param[in] column the column coordinate
     * \param[in] row the row coordinate
     */
    inline const PointT &
    operator()(size_t column, size_t row) const
    {
        return (points[row * this->width + column]);
    }

    /** \brief Obtain the point given by the (column, row) coordinates. Only works on organized
     * datasets (those that have height != 1).
     * \param[in] column the column coordinate
     * \param[in] row the row coordinate
     */
    inline PointT &
    operator()(size_t column, size_t row)
    {
        return (points[row * this->width + column]);
    }

    /** \brief Return whether a dataset is organized (e.g., arranged in a structured grid).
     * \note The height value must be different than 1 for a dataset to be organized.
     */
    inline bool
    isOrganized() const
    {
        return (height > 1);
    }

    /** \brief Return an Eigen MatrixXf (assumes float values) mapped to the specified dimensions of the PointCloud.
     * \anchor getMatrixXfMap
     * \note This method is for advanced users only! Use with care!
     *
     * \attention Since 1.4.0, Eigen matrices are forced to Row Major to increase the efficiency of the algorithms in PCL
     *   This means that the behavior of getMatrixXfMap changed, and is now correctly mapping 1-1 with a PointCloud structure,
     *   that is: number of points in a cloud = rows in a matrix, number of point dimensions = columns in a matrix
     *
     * \param[in] dim the number of dimensions to consider for each point
     * \param[in] stride the number of values in each point (will be the number of values that separate two of the columns)
     * \param[in] offset the number of dimensions to skip from the beginning of each point
     *            (stride = offset + dim + x, where x is the number of dimensions to skip from the end of each point)
     * \note for getting only XYZ coordinates out of PointXYZ use dim=3, stride=4 and offset=0 due to the alignment.
     * \attention PointT types are most of the time aligned, so the offsets are not continuous!
     */
    inline Eigen::Map<Eigen::MatrixXf, Eigen::Aligned, Eigen::OuterStride<>>
    getMatrixXfMap(int dim, int stride, int offset)
    {
        if (Eigen::MatrixXf::Flags & Eigen::RowMajorBit)
            return (Eigen::Map<Eigen::MatrixXf, Eigen::Aligned, Eigen::OuterStride<>>(reinterpret_cast<float *>(&points[0]) + offset, points.size(), dim, Eigen::OuterStride<>(stride)));
        else
            return (Eigen::Map<Eigen::MatrixXf, Eigen::Aligned, Eigen::OuterStride<>>(reinterpret_cast<float *>(&points[0]) + offset, dim, points.size(), Eigen::OuterStride<>(stride)));
    }

    /** \brief Return an Eigen MatrixXf (assumes float values) mapped to the specified dimensions of the PointCloud.
     * \anchor getMatrixXfMap
     * \note This method is for advanced users only! Use with care!
     *
     * \attention Since 1.4.0, Eigen matrices are forced to Row Major to increase the efficiency of the algorithms in PCL
     *   This means that the behavior of getMatrixXfMap changed, and is now correctly mapping 1-1 with a PointCloud structure,
     *   that is: number of points in a cloud = rows in a matrix, number of point dimensions = columns in a matrix
     *
     * \param[in] dim the number of dimensions to consider for each point
     * \param[in] stride the number of values in each point (will be the number of values that separate two of the columns)
     * \param[in] offset the number of dimensions to skip from the beginning of each point
     *            (stride = offset + dim + x, where x is the number of dimensions to skip from the end of each point)
     * \note for getting only XYZ coordinates out of PointXYZ use dim=3, stride=4 and offset=0 due to the alignment.
     * \attention PointT types are most of the time aligned, so the offsets are not continuous!
     */
    inline const Eigen::Map<const Eigen::MatrixXf, Eigen::Aligned, Eigen::OuterStride<>>
    getMatrixXfMap(int dim, int stride, int offset) const
    {
        if (Eigen::MatrixXf::Flags & Eigen::RowMajorBit)
            return (Eigen::Map<const Eigen::MatrixXf, Eigen::Aligned, Eigen::OuterStride<>>(reinterpret_cast<float *>(const_cast<PointT *>(&points[0])) + offset, points.size(), dim, Eigen::OuterStride<>(stride)));
        else
            return (Eigen::Map<const Eigen::MatrixXf, Eigen::Aligned, Eigen::OuterStride<>>(reinterpret_cast<float *>(const_cast<PointT *>(&points[0])) + offset, dim, points.size(), Eigen::OuterStride<>(stride)));
    }

    /** \brief Return an Eigen MatrixXf (assumes float values) mapped to the PointCloud.
     * \note This method is for advanced users only! Use with care!
     * \attention PointT types are most of the time aligned, so the offsets are not continuous!
     * See \ref getMatrixXfMap for more information.
     */
    inline Eigen::Map<Eigen::MatrixXf, Eigen::Aligned, Eigen::OuterStride<>>
    getMatrixXfMap()
    {
        return (getMatrixXfMap(sizeof(PointT) / sizeof(float), sizeof(PointT) / sizeof(float), 0));
    }

    /** \brief Return an Eigen MatrixXf (assumes float values) mapped to the PointCloud.
     * \note This method is for advanced users only! Use with care!
     * \attention PointT types are most of the time aligned, so the offsets are not continuous!
     * See \ref getMatrixXfMap for more information.
     */
    inline const Eigen::Map<const Eigen::MatrixXf, Eigen::Aligned, Eigen::OuterStride<>>
    getMatrixXfMap() const
    {
        return (getMatrixXfMap(sizeof(PointT) / sizeof(float), sizeof(PointT) / sizeof(float), 0));
    }

    /** \brief The point cloud header. It contains information about the acquisition time. */
    PCLHeader header;

    /** \brief The point data. */
    std::vector<PointT, Eigen::aligned_allocator<PointT>> points;

    /** \brief The point cloud width (if organized as an image-structure). */
    uint32_t width;
    /** \brief The point cloud height (if organized as an image-structure). */
    uint32_t height;

    /** \brief True if no points are invalid (e.g., have NaN or Inf values). */
    bool is_dense;

    /** \brief Sensor acquisition pose (origin/translation). */
    Eigen::Vector4f sensor_origin_;
    /** \brief Sensor acquisition pose (rotation). */
    Eigen::Quaternionf sensor_orientation_;

    typedef PointT PointType; // Make the template class available from the outside
    typedef std::vector<PointT, Eigen::aligned_allocator<PointT>> VectorType;
    typedef std::vector<PointCloud<PointT>, Eigen::aligned_allocator<PointCloud<PointT>>> CloudVectorType;
    typedef std::shared_ptr<PointCloud<PointT>> Ptr;
    typedef std::shared_ptr<const PointCloud<PointT>> ConstPtr;

    // std container compatibility typedefs according to
    // http://en.cppreference.com/w/cpp/concept/Container
    typedef PointT value_type;
    typedef PointT &reference;
    typedef const PointT &const_reference;
    typedef typename VectorType::difference_type difference_type;
    typedef typename VectorType::size_type size_type;

    // iterators
    typedef typename VectorType::iterator iterator;
    typedef typename VectorType::const_iterator const_iterator;
    inline iterator begin() { return (points.begin()); }
    inline iterator end() { return (points.end()); }
    inline const_iterator begin() const { return (points.begin()); }
    inline const_iterator end() const { return (points.end()); }

    // capacity
    inline size_t size() const { return (points.size()); }
    inline void reserve(size_t n) { points.reserve(n); }
    inline bool empty() const { return points.empty(); }

    /** \brief Resize the cloud
     * \param[in] n the new cloud size
     */
    inline void resize(size_t n)
    {
        points.resize(n);
        if (width * height != n)
        {
            width = static_cast<uint32_t>(n);
            height = 1;
        }
    }

    // element access
    inline const PointT &operator[](size_t n) const { return (points[n]); }
    inline PointT &operator[](size_t n) { return (points[n]); }
    inline const PointT &at(size_t n) const { return (points.at(n)); }
    inline PointT &at(size_t n) { return (points.at(n)); }
    inline const PointT &front() const { return (points.front()); }
    inline PointT &front() { return (points.front()); }
    inline const PointT &back() const { return (points.back()); }
    inline PointT &back() { return (points.back()); }

    /** \brief Insert a new point in the cloud, at the end of the container.
     * \note This breaks the organized structure of the cloud by setting the height to 1!
     * \param[in] pt the point to insert
     */
    inline void
    push_back(const PointT &pt)
    {
        points.push_back(pt);
        width = static_cast<uint32_t>(points.size());
        height = 1;
    }

    /** \brief Insert a new point in the cloud, given an iterator.
     * \note This breaks the organized structure of the cloud by setting the height to 1!
     * \param[in] position where to insert the point
     * \param[in] pt the point to insert
     * \return returns the new position iterator
     */
    inline iterator
    insert(iterator position, const PointT &pt)
    {
        iterator it = points.insert(position, pt);
        width = static_cast<uint32_t>(points.size());
        height = 1;
        return (it);
    }

    /** \brief Insert a new point in the cloud N times, given an iterator.
     * \note This breaks the organized structure of the cloud by setting the height to 1!
     * \param[in] position where to insert the point
     * \param[in] n the number of times to insert the point
     * \param[in] pt the point to insert
     */
    inline void
    insert(iterator position, size_t n, const PointT &pt)
    {
        points.insert(position, n, pt);
        width = static_cast<uint32_t>(points.size());
        height = 1;
    }

    /** \brief Insert a new range of points in the cloud, at a certain position.
     * \note This breaks the organized structure of the cloud by setting the height to 1!
     * \param[in] position where to insert the data
     * \param[in] first where to start inserting the points from
     * \param[in] last where to stop inserting the points from
     */
    template <class InputIterator>
    inline void
    insert(iterator position, InputIterator first, InputIterator last)
    {
        points.insert(position, first, last);
        width = static_cast<uint32_t>(points.size());
        height = 1;
    }

    /** \brief Erase a point in the cloud.
     * \note This breaks the organized structure of the cloud by setting the height to 1!
     * \param[in] position what data point to erase
     * \return returns the new position iterator
     */
    inline iterator
    erase(iterator position)
    {
        iterator it = points.erase(position);
        width = static_cast<uint32_t>(points.size());
        height = 1;
        return (it);
    }

    /** \brief Erase a set of points given by a (first, last) iterator pair
     * \note This breaks the organized structure of the cloud by setting the height to 1!
     * \param[in] first where to start erasing points from
     * \param[in] last where to stop erasing points from
     * \return returns the new position iterator
     */
    inline iterator
    erase(iterator first, iterator last)
    {
        iterator it = points.erase(first, last);
        width = static_cast<uint32_t>(points.size());
        height = 1;
        return (it);
    }

    /** \brief Swap a point cloud with another cloud.
     * \param[in,out] rhs point cloud to swap this with
     */
    inline void
    swap(PointCloud<PointT> &rhs)
    {
        this->points.swap(rhs.points);
        std::swap(width, rhs.width);
        std::swap(height, rhs.height);
        std::swap(is_dense, rhs.is_dense);
        std::swap(sensor_origin_, rhs.sensor_origin_);
        std::swap(sensor_orientation_, rhs.sensor_orientation_);
    }

    /** \brief Removes all points in a cloud and sets the width and height to 0. */
    inline void
    clear()
    {
        points.clear();
        width = 0;
        height = 0;
    }

    /** \brief Copy the cloud to the heap and return a smart pointer
     * Note that deep copy is performed, so avoid using this function on non-empty clouds.
     * The changes of the returned cloud are not mirrored back to this one.
     * \return shared pointer to the copy of the cloud
     */
    inline Ptr
    makeShared() const { return Ptr(new PointCloud<PointT>(*this)); }

protected:
    // This is motivated by ROS integration. Users should not need to access mapping_.
    std::shared_ptr<MsgFieldMap> mapping_;

    friend std::shared_ptr<MsgFieldMap> &getMapping<PointT>(PointCloud<PointT> &p);

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <typename PointT>
std::shared_ptr<MsgFieldMap> & getMapping(PointCloud<PointT> &p)
{
    return (p.mapping_);
}

typedef PointXYZINormal PointType;
typedef vector<PointType, Eigen::aligned_allocator<PointType>>  PointVector;
typedef PointCloud<PointType> PointCloudXYZINormal;
#define NUM_MATCH_POINTS (5)
const PointType ZeroP;

struct KD_TREE_NODE
{
    PointType point;
    int division_axis;  
    int TreeSize = 1;
    int invalid_point_num = 0;
    int down_del_num = 0;
    bool point_deleted = false;
    bool tree_deleted = false; 
    bool point_downsample_deleted = false;
    bool tree_downsample_deleted = false;
    bool need_push_down_to_left = false;
    bool need_push_down_to_right = false;
    bool working_flag = false;
    pthread_mutex_t push_down_mutex_lock;
    float node_range_x[2], node_range_y[2], node_range_z[2];   
    KD_TREE_NODE *left_son_ptr = nullptr;
    KD_TREE_NODE *right_son_ptr = nullptr;
    KD_TREE_NODE *father_ptr = nullptr;
    // For paper data record
    float alpha_del;
    float alpha_bal;
};

struct PointType_CMP{
    PointType point;
    float dist = 0.0;
    PointType_CMP (PointType p = ZeroP, float d = INFINITY){
        this->point = p;
        this->dist = d;
    };
    bool operator < (const PointType_CMP &a)const{
        if (fabs(dist - a.dist) < 1e-10) return point.x < a.point.x;
        else return dist < a.dist;
    }
    //* 测试： 改变一下就可以用来处理最小值的情况了，下面这种情况对应第一个总是最小值
    // bool operator < (const PointType_CMP &a)const{
    //     if (fabs(dist - a.dist) < 1e-10) return point.x > a.point.x;
    //     else return dist > a.dist;
    // }
};

struct BoxPointType{
    float vertex_min[3];
    float vertex_max[3];
};

enum operation_set {ADD_POINT, DELETE_POINT, DELETE_BOX, ADD_BOX, DOWNSAMPLE_DELETE, PUSH_DOWN};

enum delete_point_storage_set {NOT_RECORD, DELETE_POINTS_REC, MULTI_THREAD_REC};

struct Operation_Logger_Type{
    PointType point;
    BoxPointType boxpoint;
    bool tree_deleted, tree_downsample_deleted;
    operation_set op;
};

class MANUAL_Q{
    private:
        int head = 0,tail = 0, counter = 0;
        Operation_Logger_Type q[Q_LEN];
        bool is_empty;
    public:
        void pop();
        Operation_Logger_Type front();
        Operation_Logger_Type back();
        void clear();
        void push(Operation_Logger_Type op);
        bool empty();
        int size();
};

class MANUAL_HEAP
{
    public:
        MANUAL_HEAP(int max_capacity = 100);
        ~MANUAL_HEAP();
        void pop();
        PointType_CMP top();
        void push(PointType_CMP point);
        int size();
        void clear();
        void print()
        {
            for(int i=0;i<heap_size;i++)
                cout<<heap[i].dist<<", ";
            cout<<endl;
        }
    private:
        PointType_CMP * heap;
        void MoveDown(int heap_index);
        void FloatUp(int heap_index);
        int heap_size = 0;
        int cap = 0;
};


class KD_TREE
{
public:
    // Multi-thread Tree Rebuild
    bool termination_flag = false;
    bool rebuild_flag = false;
    pthread_t rebuild_thread;
    pthread_mutex_t termination_flag_mutex_lock, rebuild_ptr_mutex_lock, working_flag_mutex, search_flag_mutex;
    pthread_mutex_t rebuild_logger_mutex_lock, points_deleted_rebuild_mutex_lock;
    // queue<Operation_Logger_Type> Rebuild_Logger;
    MANUAL_Q Rebuild_Logger;    
    PointVector Rebuild_PCL_Storage;
    KD_TREE_NODE ** Rebuild_Ptr = nullptr;
    int search_mutex_counter = 0;
    static void * multi_thread_ptr(void *arg);
    void multi_thread_rebuild();
    void start_thread();
    void stop_thread();
    void run_operation(KD_TREE_NODE ** root, Operation_Logger_Type operation);
    // KD Tree Functions and augmented variables
    int Treesize_tmp = 0, Validnum_tmp = 0;
    float alpha_bal_tmp = 0.5, alpha_del_tmp = 0.0;
    float delete_criterion_param = 0.5f;
    float balance_criterion_param = 0.7f;
    float downsample_size = 0.2f;
    bool Delete_Storage_Disabled = false;
    KD_TREE_NODE * STATIC_ROOT_NODE = nullptr;
    PointVector Points_deleted;
    PointVector Downsample_Storage;
    PointVector Multithread_Points_deleted;
    void InitTreeNode(KD_TREE_NODE * root);
    void Test_Lock_States(KD_TREE_NODE *root);
    void BuildTree(KD_TREE_NODE ** root, int l, int r, PointVector & Storage);
    void Rebuild(KD_TREE_NODE ** root);
    int Delete_by_range(KD_TREE_NODE ** root, BoxPointType boxpoint, bool allow_rebuild, bool is_downsample);
    void Delete_by_point(KD_TREE_NODE ** root, PointType point, bool allow_rebuild);
    void Add_by_point(KD_TREE_NODE ** root, PointType point, bool allow_rebuild, int father_axis);
    void Add_by_range(KD_TREE_NODE ** root, BoxPointType boxpoint, bool allow_rebuild);
    void Search(KD_TREE_NODE * root, int k_nearest, PointType point, MANUAL_HEAP &q, double max_dist);//priority_queue<PointType_CMP>
    void Search_by_range(KD_TREE_NODE *root, BoxPointType boxpoint, PointVector &Storage);
    bool Criterion_Check(KD_TREE_NODE * root);
    void Push_Down(KD_TREE_NODE * root);
    void Update(KD_TREE_NODE * root); 
    void delete_tree_nodes(KD_TREE_NODE ** root);
    void downsample(KD_TREE_NODE ** root);
    bool same_point(PointType a, PointType b);
    float calc_dist(PointType a, PointType b);
    float calc_box_dist(KD_TREE_NODE * node, PointType point);    
    static bool point_cmp_x(PointType a, PointType b); 
    static bool point_cmp_y(PointType a, PointType b); 
    static bool point_cmp_z(PointType a, PointType b); 
    void print_treenode(KD_TREE_NODE * root, int index, FILE *fp, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max);
    

public:
    KD_TREE(float delete_param = 0.5, float balance_param = 0.6 , float box_length = 0.2);
    ~KD_TREE();
    void Set_delete_criterion_param(float delete_param);
    void Set_balance_criterion_param(float balance_param);
    void set_downsample_param(float box_length);
    void InitializeKDTree(float delete_param = 0.5, float balance_param = 0.7, float box_length = 0.2); 
    int size();
    int validnum();
    void root_alpha(float &alpha_bal, float &alpha_del);
    void Build(PointVector point_cloud);
    void Nearest_Search(PointType point, int k_nearest, PointVector &Nearest_Points, vector<float> & Point_Distance, double max_dist = INFINITY);
    int Add_Points(PointVector & PointToAdd, bool downsample_on);
    void Add_Point_Boxes(vector<BoxPointType> & BoxPoints);
    void Delete_Points(PointVector & PointToDel);
    int Delete_Point_Boxes(vector<BoxPointType> & BoxPoints);
    void flatten(KD_TREE_NODE * root, PointVector &Storage, delete_point_storage_set storage_type);
    void acquire_removed_points(PointVector & removed_points);
    void print_tree(int index, FILE *fp, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max);
    BoxPointType tree_range();
    void myreset();
    PointVector PCL_Storage;     
    KD_TREE_NODE * Root_Node = nullptr;
    int max_queue_size = 0;
};
