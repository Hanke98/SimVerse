#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>

#include "RobotArm.h"
#include "DataTypes.h"
#include "UrdfParser.h"
#include "../../../../src/Core/Matrix/Transform3x3.h"

namespace py = pybind11;
using namespace dyno;

// ----------------- 常用类型别名 -----------------

using DataType   = DataType3f;
using RobotArm   = RobotArmSimulator<DataType>;

using Real       = typename DataType::Real;
using Coord      = typename DataType::Coord;
using Vec3       = Vec3f;
using QuatType   = typename RobotArm::TQuat;
using Matrix     = typename DataType::Matrix;     // Mat3f
using TransformT = Transform3f;

// Param 类型（来自 RobotArm.h 中的 typedef）
using CtrlParam        = typename RobotArm::CtrlParam;
using HingeTorqueParam = typename RobotArm::HingeTorqueParam;
using LocalIndexParam  = typename RobotArm::LocalIndexParam;
using MassParam        = typename RobotArm::MassParam;
using InertiaParam     = typename RobotArm::InertiaParam;
using ResetParam       = typename RobotArm::ResetParam;

// URDF 相关
using UrdfInformation  = dyno::UrdfInformation;
using UrdfLink         = dyno::UrdfLink;
using UrdfJointLimits  = dyno::UrdfJointLimits;
using UrdfJoint        = dyno::UrdfJoint;


// =====================================================
// Vec3f 绑定
// =====================================================

void bindVec3(py::module_ &m) {
    py::class_<Vec3>(m, "Vec3f")
        .def(py::init<float, float, float>(),
             py::arg("x") = 0.f,
             py::arg("y") = 0.f,
             py::arg("z") = 0.f)
        .def_readwrite("x", &Vec3::x)
        .def_readwrite("y", &Vec3::y)
        .def_readwrite("z", &Vec3::z)
        .def("__repr__", [](const Vec3 &v) {
            return "Vec3f(" +
                   std::to_string(v.x) + ", " +
                   std::to_string(v.y) + ", " +
                   std::to_string(v.z) + ")";
        });
}


// =====================================================
// TQuat 绑定（含 inverse / normalize / rotate / toRotationAxis）
// =====================================================

void bindQuat(py::module_ &m) {
    py::class_<QuatType>(m, "TQuat")
        // 构造：Quat(Real x, Real y, Real z, Real w)
        .def(py::init<Real, Real, Real, Real>(),
             py::arg("x") = 0,
             py::arg("y") = 0,
             py::arg("z") = 0,
             py::arg("w") = 1)

        .def_readwrite("w", &QuatType::w)
        .def_readwrite("x", &QuatType::x)
        .def_readwrite("y", &QuatType::y)
        .def_readwrite("z", &QuatType::z)

        // 归一化（原地）
        .def("normalize",
             &QuatType::normalize,
             "Normalize the quaternion in-place and return self")

        // 求逆
        .def("inverse",
             &QuatType::inverse,
             "Return the inverse of the quaternion")

        // 旋转向量：Vector<Real,3> rotate(const Vector<Real,3>& v) const;
        .def("rotate",
             &QuatType::rotate,
             py::arg("v"),
             "Rotate a Vec3f by this quaternion")

        // 轴角表示：void toRotationAxis(Real &rot, Vector<Real,3> &axis) const;
        // Python 返回 (rot, axis: Vec3f)
        .def("to_rotation_axis",
             [](const QuatType& q) {
                 Real rot;
                 dyno::Vector<Real, 3> axis;
                 q.toRotationAxis(rot, axis);
                 // 假设 Vector<Real,3> 和 Vec3f 是兼容的
                 return py::make_tuple(rot, Vec3(axis[0], axis[1], axis[2]));
             },
             "Return (angle, axis) where angle is in radians")

        .def("__mul__", [](const QuatType& a, const QuatType& b) {
                return a * b;
             }, py::is_operator())
        .def("__repr__", [](const QuatType &q) {
            return "TQuat(" +
                   std::to_string(q.x) + ", " +
                   std::to_string(q.y) + ", " +
                   std::to_string(q.z) + ", " +
                   std::to_string(q.w) + ")";
        });
}


// =====================================================
// Mat3f / Transform3f 绑定
// =====================================================

void bindMatrix(py::module_ &m) {
    // Mat3f = SquareMatrix<float,3>
    py::class_<Matrix>(m, "Mat3f")
        .def(py::init<>())
        .def(py::init<float>(),
             py::arg("diag_value"))
        .def(py::init<
                 float, float, float,
                 float, float, float,
                 float, float, float>(),
             py::arg("m00"), py::arg("m01"), py::arg("m02"),
             py::arg("m10"), py::arg("m11"), py::arg("m12"),
             py::arg("m20"), py::arg("m21"), py::arg("m22"))
        .def_static("rows",  &Matrix::rows)
        .def_static("cols",  &Matrix::cols)
        .def("determinant",  &Matrix::determinant)
        .def("trace",        &Matrix::trace)
        .def("__getitem__", [](const Matrix &mat, std::pair<int,int> idx) {
            return mat(static_cast<unsigned int>(idx.first),
                       static_cast<unsigned int>(idx.second));
        })
        .def("__setitem__", [](Matrix &mat, std::pair<int,int> idx, float value) {
            mat(static_cast<unsigned int>(idx.first),
                static_cast<unsigned int>(idx.second)) = value;
        })
        .def("__repr__", [](const Matrix &mat) {
            std::ostringstream oss;
            oss << "Mat3f([\n";
            for (int i = 0; i < 3; ++i) {
                oss << "  ";
                for (int j = 0; j < 3; ++j) {
                    oss << mat(i,j);
                    if (j < 2) oss << ", ";
                }
                oss << "\n";
            }
            oss << "])";
            return oss.str();
        });
}

void bindTransform(py::module_ &m) {
    py::class_<TransformT>(m, "Transform3f")
        .def(py::init<>())
        .def(py::init<const Vec3&, const Matrix&, const Vec3&>(),
             py::arg("translation"),
             py::arg("rotation"),
             py::arg("scale") = Vec3(1.f, 1.f, 1.f))
        .def_property(
            "translation",
            [](TransformT &t) { return t.translation(); },
            [](TransformT &t, const Vec3 &v) { t.translation() = v; }
        )
        .def_property(
            "rotation",
            [](TransformT &t) { return t.rotation(); },
            [](TransformT &t, const Matrix &m_) { t.rotation() = m_; }
        )
        .def_property(
            "scale",
            [](TransformT &t) { return t.scale(); },
            [](TransformT &t, const Vec3 &v) { t.scale() = v; }
        )
        .def("__mul__", [](const TransformT &t, const Vec3 &v) {
            return t * v;
        }, py::is_operator())
        .def("__repr__", [](const TransformT &t) {
            std::ostringstream oss;
            auto tr = t.translation();
            oss << "Transform3f(translation=("
                << tr.x << ", " << tr.y << ", " << tr.z << "), ...)";
            return oss.str();
        });
}


// =====================================================
// BatchRigidBodySystem 参数结构绑定
// =====================================================

void bindParams(py::module_ &m) {
    // 基类：BatchRigidBodySystemControlParamBase
    py::class_<CtrlParam>(m, "CtrlParam")
        .def(py::init<>())
        .def_readwrite("num_bodies", &CtrlParam::num_bodies)
        .def_readwrite("ids",        &CtrlParam::ids);

    // ResetParam
    py::class_<ResetParam, CtrlParam>(m, "ResetParam")
        .def(py::init<>())
        .def_readwrite("targetPosition", &ResetParam::targetPosition);
        // targetPosition: std::vector<Coord> => std::vector<Vec3f>

    // HingeTorqueParam
    py::class_<HingeTorqueParam, CtrlParam>(m, "HingeTorqueParam")
        .def(py::init<>())
        .def_readwrite("torques", &HingeTorqueParam::torques);
        // torques: std::vector<std::vector<float>>

    // LocalIndexParam
    py::class_<LocalIndexParam, CtrlParam>(m, "LocalIndexParam")
        .def(py::init<>())
        .def_readwrite("localRigidBodyid", &LocalIndexParam::localRigidBodyid);

    // MassParam
    py::class_<MassParam, CtrlParam>(m, "MassParam")
        .def(py::init<>())
        .def_readwrite("mass", &MassParam::mass);
        // mass: std::vector<std::vector<float>>

    // InertiaParam
    py::class_<InertiaParam, CtrlParam>(m, "InertiaParam")
        .def(py::init<>())
        .def_readwrite("inertia", &InertiaParam::inertia);
        // inertia: std::vector<std::vector<Matrix>> => Mat3f
}


// =====================================================
// URDF 信息结构绑定
// =====================================================

void bindUrdf(py::module_ &m) {
    py::class_<UrdfJointLimits>(m, "UrdfJointLimits")
        .def(py::init<>())
        .def_readwrite("lower",    &UrdfJointLimits::lower)
        .def_readwrite("upper",    &UrdfJointLimits::upper)
        .def_readwrite("effort",   &UrdfJointLimits::effort)
        .def_readwrite("velocity", &UrdfJointLimits::velocity);

    py::class_<UrdfLink>(m, "UrdfLink")
        .def(py::init<>())
        .def_readwrite("name",             &UrdfLink::name)
        .def_readwrite("visualMeshPath",   &UrdfLink::visualMeshPath)
        .def_readwrite("meshTransform",    &UrdfLink::meshTransform)  // Transform3f
        .def_readwrite("collisionMeshPath",&UrdfLink::collisionMeshPath)
        .def_readwrite("T_world",          &UrdfLink::T_world)        // Transform3f
        .def_readwrite("shapeId",          &UrdfLink::shapeId)
        .def_readwrite("isRoot",           &UrdfLink::isRoot);

    py::class_<UrdfJoint>(m, "UrdfJoint")
        .def(py::init<>())
        .def_readwrite("name",         &UrdfJoint::name)
        // type (UrdfJointType) 暂不暴露
        .def_readwrite("parentLink",   &UrdfJoint::parentLink)
        .def_readwrite("childLink",    &UrdfJoint::childLink)
        .def_readwrite("parentLinkId", &UrdfJoint::parentLinkId)
        .def_readwrite("childLinkId",  &UrdfJoint::childLinkId)
        .def_readwrite("axis",         &UrdfJoint::axis)
        .def_readwrite("axisWorld",    &UrdfJoint::axisWorld)
        .def_readwrite("originLocal",  &UrdfJoint::originLocal)   // Transform3f
        .def_readwrite("originWorld",  &UrdfJoint::originWorld)   // Transform3f
        .def_readwrite("limits",       &UrdfJoint::limits)
        .def_readwrite("damping",      &UrdfJoint::damping);

    py::class_<UrdfInformation>(m, "UrdfInformation")
        .def(py::init<>())
        .def_readwrite("links",     &UrdfInformation::links)
        .def_readwrite("joints",    &UrdfInformation::joints)
        .def_readwrite("robotName", &UrdfInformation::robotName);
}


// =====================================================
// RobotArmSimulator 绑定
// =====================================================

void bindRobotArmSimulator(py::module_ &m) {
    py::class_<RobotArm>(m, "RobotArmSimulator")
        .def(py::init<>())

        // ---------- 场景 / 仿真控制 ----------
        .def("initBatchSolver",
             &RobotArm::initBatchSolver,
             "初始化 batch solver")

        .def("addRobotArmRigidBodies",
             &RobotArm::addRobotArmRigidBodies,
             py::arg("urdf_fn"),
             py::arg("density")         = 1000.0f,
             py::arg("target_position") = std::vector<Vec3>{},
             py::arg("render_boundingbox") = true,
             "从 URDF 文件添加机械臂刚体系统")

        .def("resetStates",
             &RobotArm::resetStates,
             py::arg("param"),
             "通过 ResetParam 重置状态")

        .def("resetTargets",
                 &RobotArm::resetTargets,
                 py::arg("param"),
                 "通过 ResetParam 重置状态")

        .def("createScene",
             &RobotArm::createScene,
             "创建场景")

        .def("setupSceneGraph",
             &RobotArm::setupSceneGraph,
             "搭建场景图")

        .def("initialize",
             &RobotArm::initialize,
             py::arg("width")  = 1280,
             py::arg("height") = 768,
             py::arg("scale")  = 1.0f,
             "初始化仿真窗口和渲染")

        .def("stepSimulation",
             &RobotArm::stepSimulation,
             py::arg("enableRendering") = true,
             "执行一步仿真（可选是否渲染）")

        .def("terminateSimulation",
             &RobotArm::terminateSimulation,
             "终止仿真/关闭窗口")


        // ---------- Getters ----------
        .def("getCentersByLocalIndex",
             &RobotArm::getCentersByLocalIndex,
             py::arg("param"),
             "根据 LocalIndexParam 获取刚体质心位置列表")

        .def("getVelocitiesByLocalIndex",
             &RobotArm::getVelocitiesByLocalIndex,
             py::arg("param"),
             "根据 LocalIndexParam 获取刚体线速度列表")

        .def("getAnglesByLocalIndex",
             &RobotArm::getAnglesByLocalIndex,
             py::arg("param"),
             "根据 LocalIndexParam 获取刚体姿态四元数列表")

        .def("getAnglesVectorByLocalIndex",
             &RobotArm::getAnglesVectorByLocalIndex,
             py::arg("param"),
             "根据 LocalIndexParam 获取关节角（向量形式）")

        .def("getAngularVelocitiesByLocalIndex",
             &RobotArm::getAngularVelocitiesByLocalIndex,
             py::arg("param"),
             "根据 LocalIndexParam 获取角速度")

        .def("getMassByLocalIndex",
             &RobotArm::getMassByLocalIndex,
             py::arg("param"),
             "根据 LocalIndexParam 获取质量列表")

        .def("getKinematicsChainInfo",
             &RobotArm::getKinematicsChainInfo,
             "获取 URDF 运动学链信息（UrdfInformation）")

        .def("getTransform",
             &RobotArm::getTransform,
             py::arg("param"),
             "获取 URDF 运动学链信息（UrdfInformation）")


        // ---------- Setters / 控制接口 ----------
        .def("setAngularDamping",
             &RobotArm::setAngularDamping,
             py::arg("damping"),
             "设置角阻尼系数")

        .def("setDt",
             &RobotArm::setDt,
             py::arg("dt"),
             "设置仿真时间步长")

        .def("enableGravity",
             &RobotArm::enableGravity,
             py::arg("flag"),
             "启用/关闭重力")

        .def("enableFriction",
             &RobotArm::enableFriction,
             py::arg("flag"),
             "启用/关闭摩擦")

        .def("setHingeTorques",
             &RobotArm::setHingeTorques,
             py::arg("hinge_torque_param"),
             "设置铰接关节扭矩控制参数")

        .def("setMass",
             &RobotArm::setMass,
             py::arg("mass_param"),
             "设置质量参数")

        .def("setInertia",
             &RobotArm::setInertia,
             py::arg("inertia_param"),
             "设置惯性参数")

        .def("setTransform",
             &RobotArm::setTransform,
             py::arg("base"),
             py::arg("offset"),
             py::arg("num_copies_x"),
             py::arg("num_copies_y"),
             py::arg("num_copies_z"),
             "设置机械臂阵列在空间中的布局");
}


// =====================================================
// 模块入口
// =====================================================

PYBIND11_MODULE(RobotArm_pybind, m) {
    m.doc() = "RobotArmSimulator (DataType3f) pybind11 bindings";

    bindVec3(m);
    bindQuat(m);
    bindMatrix(m);
    bindTransform(m);
    bindParams(m);
    bindUrdf(m);
    bindRobotArmSimulator(m);
}