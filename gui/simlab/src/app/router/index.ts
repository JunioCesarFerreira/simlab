import { createRouter, createWebHistory, type RouteRecordRaw } from "vue-router";

const routes: RouteRecordRaw[] = [
  { path: "/", redirect: "/dashboard" },
  { path: "/dashboard", component: () => import("../../pages/Dashboard.vue") },
  { path: "/experiments", component: () => import("../../pages/ExperimentsList.vue") },
  {
    path: "/experiments/:id",
    component: () => import("../../pages/ExperimentDetail.vue"),
    props: true,
  },
];

export default createRouter({
  history: createWebHistory(),
  routes,
});
